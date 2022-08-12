from cgi import test
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm


from arguments import get_parser
from setup import setup_r1_nes
from ttp.ttp_dataset import read_prob, prob_to_env
from policy.r1_nes import ExperienceReplay, R1_NES
from utils import solve, write_test_phn_progress, save_nes

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def train_one_epoch(agent, policy: R1_NES, train_env, writer, pop_size=10, max_iter=20):
    agent.eval()
    er = ExperienceReplay(dim=policy.n_params, num_obj=2, num_sample=pop_size)
    for it in tqdm(range(max_iter)):
        travel_time_list = torch.zeros((pop_size, 1), dtype=torch.float32)
        total_profit_list = torch.zeros((pop_size, 1), dtype=torch.float32)
        node_order_list = torch.zeros((pop_size, train_env.num_nodes), dtype=torch.long)
        item_selection_list = torch.zeros((pop_size, train_env.num_items), dtype=torch.bool)
        param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
        for n, param_dict in enumerate(param_dict_list):
            tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, train_env, param_dict, normalized=True)
            node_order_list[n] = tour_list
            item_selection_list[n] = item_selection
            travel_time_list[n] = tour_lengths
            total_profit_list[n] = total_profits

        inv_total_profit_list = 1 - total_profit_list
        f_list = torch.cat((inv_total_profit_list, travel_time_list), dim=1)
        er.add(policy, sample_list, f_list, node_order_list, item_selection_list)
        if er.num_saved_policy < er.max_saved_policy:
            continue
        policy.update_with_er(er)

@torch.no_grad()
def test_one_epoch(agent, policy, test_env, writer, epoch, pop_size=100):
    agent.eval()
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    solution_list = []
    for n, param_dict in enumerate(param_dict_list):    
        tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve(agent, test_env, param_dict, normalized=False)
        solution_list += [torch.stack([tour_length, total_profit], dim=1)]
    solution_list = torch.cat(solution_list)
    write_test_phn_progress(writer, solution_list, epoch)

def run(args):
    agent, policy, last_epoch, writer, checkpoint_path, test_env = setup_r1_nes(args)
    training_size = args.num_training_samples
    num_nodes_list = [50]
    num_items_per_city_list = [1,3,5]
    config_list = [(num_nodes, num_items_per_city, idx) for num_nodes in num_nodes_list for num_items_per_city in num_items_per_city_list for idx in range(1000)]
    num_configs = len(num_nodes_list)*len(num_items_per_city_list)
    for epoch in range(last_epoch, args.max_epoch):
        print("EPOCH:", epoch)
        print("---------------------------------------")
        config_it = epoch%num_configs
        if config_it == 0:
            random.shuffle(config_list)
        num_nodes, num_items_per_city, prob_idx = config_list[config_it]
        train_env = prob_to_env(read_prob(num_nodes, num_items_per_city, prob_idx))
        train_one_epoch(agent, policy, train_env, writer)
        # # validation_cost = validation_one_epoch(agent, validation_dataset, writer)
        test_one_epoch(agent, policy, test_env, writer, epoch)
        save_nes(policy, epoch, checkpoint_path)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    args = prepare_args()
    torch.set_num_threads(os.cpu_count())
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)