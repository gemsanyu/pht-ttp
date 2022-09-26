import math
from multiprocessing import Pool
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from arguments import get_parser
from setup import setup_r1_nes
from ttp.ttp_dataset import read_prob, prob_to_env
from ttp.ttp import TTP
from ttp.utils import save_prob
from policy.utils import update_nondom_archive
from policy.snes import ExperienceReplay, SNES
from utils import write_test_phn_progress, save_nes
from transformer_utils import solve

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args


@torch.no_grad()
def train_one_epoch(agent, policy: SNES, train_prob: TTP, writer, step, pop_size=10, max_saved_policy=5, max_iter=20):
    agent.eval()
    if policy.batch_size is not None:
        pop_size = int(math.ceil(policy.batch_size/max_saved_policy))
    er = ExperienceReplay(dim=policy.n_params, num_obj=2, max_saved_policy=max_saved_policy, num_sample=pop_size)
    # keep worst points
    train_env = prob_to_env(train_prob)
    for it in tqdm(range(max_iter)):
        param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
        travel_time_list = torch.zeros((pop_size, 1), dtype=torch.float32)
        total_profit_list = torch.zeros((pop_size, 1), dtype=torch.float32)
        node_order_list = torch.zeros((pop_size, train_env.num_nodes), dtype=torch.long)
        item_selection_list = torch.zeros((pop_size, train_env.num_items), dtype=torch.bool)
        for n, param_dict in enumerate(param_dict_list):
            tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, train_env, param_dict)
            node_order_list[n] = tour_list
            item_selection_list[n] = item_selection
            travel_time_list[n] = tour_lengths
            total_profit_list[n] = total_profits

        travel_time_list = travel_time_list/train_env.best_route_length_tsp -1
        inv_total_profit_list = 1 - total_profit_list/train_env.best_profit_kp
        max_curr_f1 = torch.max(inv_total_profit_list).unsqueeze(0)
        max_curr_f2 = torch.max(travel_time_list).unsqueeze(0)
        max_curr_f = torch.cat([max_curr_f1, max_curr_f2])
        train_prob.reference_point = torch.maximum(train_prob.reference_point, max_curr_f)
        f_list = torch.cat((inv_total_profit_list, travel_time_list), dim=1)
        train_prob.nondom_archive = update_nondom_archive(train_prob.nondom_archive, f_list)
        er.add(policy, sample_list, f_list)
        step += 1
        if er.num_saved_policy < er.max_saved_policy:
            continue
        policy.update_with_er(er, train_prob.reference_point, train_prob.nondom_archive)
        policy.write_progress_to_tb(writer, step)

    return step, train_prob

@torch.no_grad()
def test_one_epoch(agent, policy, test_env, sample_solutions, writer, epoch, pop_size=50):
    agent.eval()
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    solution_list = []
    for n, param_dict in enumerate(param_dict_list):
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, test_env, param_dict)
        solution_list += [torch.stack([tour_lengths, total_profits], dim=1)]
    solution_list = torch.cat(solution_list)
    write_test_phn_progress(writer, solution_list, epoch, sample_solutions)

def run(args):
    agent, policy, last_epoch, writer, checkpoint_path, test_env, sample_solutions = setup_r1_nes(args)
    num_nodes_list = [50]
    num_items_per_city_list = [1,3,5]
    config_list = [(num_nodes, num_items_per_city, idx) for num_nodes in num_nodes_list for num_items_per_city in num_items_per_city_list for idx in range(5)]
    num_configs = len(num_nodes_list)*len(num_items_per_city_list)
    step=1
    # process_pool = Pool(processes=6)

    for epoch in range(last_epoch, args.max_epoch):
        config_it = epoch%num_configs
        if config_it == 0:  
            random.shuffle(config_list)
        num_nodes, num_items_per_city, prob_idx = config_list[config_it]
        print("EPOCH:", epoch, "NN:", num_nodes, "NIC:", num_items_per_city)
        train_prob = read_prob(num_nodes, num_items_per_city, prob_idx)
        step, updated_train_prob = train_one_epoch(agent, policy, train_prob, writer, step)
        if updated_train_prob is not None:
            save_prob(updated_train_prob, num_nodes, num_items_per_city, prob_idx)
        test_one_epoch(agent, policy, test_env, sample_solutions, writer, epoch)
        save_nes(policy, epoch, checkpoint_path)

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(4)
    # torch.set_num_threads()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)