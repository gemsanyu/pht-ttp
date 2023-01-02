import math
from multiprocessing import Pool
import subprocess
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from agent.agent import Agent
from arguments import get_parser
from setup import setup_r1_nes
from ttp.ttp_dataset import read_prob, prob_to_env
from ttp.ttp import TTP
from ttp.utils import save_prob
from policy.utils import update_nondom_archive
from policy.r1_nes import R1_NES, ExperienceReplay
from utils import save_nes, solve_decode_only
from validate_r1nes import test_one_epoch

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def train_one_epoch(agent:Agent, policy: R1_NES, train_prob: TTP, writer, step, pop_size=10, max_saved_policy=5, max_iter=20):
    agent.eval()
    if policy.batch_size is not None:
        pop_size = int(math.ceil(policy.batch_size/max_saved_policy))
    er = ExperienceReplay(dim=policy.n_params, num_obj=2, max_saved_policy=max_saved_policy, num_sample=pop_size)
    train_env = prob_to_env(train_prob)
    
    # encode/embed first, it can be reused for same env/problem
    static_features = train_env.get_static_features()
    static_features = torch.from_numpy(static_features).to(agent.device)
    item_init_embed = agent.item_init_embedder(static_features[:, :train_env.num_items, :])
    depot_init_embed = agent.depot_init_embed.expand(size=(train_env.batch_size,1,-1))
    node_init_embed = agent.node_init_embed.expand(size=(train_env.batch_size,train_env.num_nodes-1,-1))
    init_embed = torch.cat([item_init_embed, depot_init_embed, node_init_embed], dim=1)
    static_embeddings, graph_embeddings = agent.gae(init_embed)
    fixed_context = agent.project_fixed_context(graph_embeddings)
    glimpse_K_static, glimpse_V_static, logits_K_static = agent.project_embeddings(static_embeddings).chunk(3, dim=-1)
    glimpse_K_static = agent._make_heads(glimpse_K_static)
    glimpse_V_static = agent._make_heads(glimpse_V_static)
    pop_size = policy.batch_size
    for it in tqdm(range(max_iter)):
        param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
        travel_time_list = torch.zeros((pop_size, 1), dtype=torch.float32)
        total_profit_list = torch.zeros((pop_size, 1), dtype=torch.float32)
        node_order_list = torch.zeros((pop_size, train_env.num_nodes), dtype=torch.long)
        item_selection_list = torch.zeros((pop_size, train_env.num_items), dtype=torch.bool)        

        for n, param_dict in enumerate(param_dict_list):
            solve_output = solve_decode_only(agent, train_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
            tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
            node_order_list[n] = tour_list
            item_selection_list[n] = item_selection
            travel_time_list[n] = tour_lengths
            total_profit_list[n] = total_profits

        inv_total_profit_list = -total_profit_list
        f_list = torch.cat((inv_total_profit_list, travel_time_list), dim=1)
        # max_curr_f1 = torch.max(travel_time_list).unsqueeze(0)
        # max_curr_f2 = torch.max(inv_total_profit_list).unsqueeze(0)
        # max_curr_f = torch.cat([max_curr_f1, max_curr_f2])
        # train_prob.reference_point = torch.maximum(train_prob.reference_point, max_curr_f)
        # train_prob.nondom_archive = update_nondom_archive(train_prob.nondom_archive, f_list)
        # er.add(policy, sample_list, f_list, node_order_list, item_selection_list)
        step += 1
        x_list = sample_list - policy.mu
        w_list = x_list/torch.exp(policy.ld)
        policy.update(w_list, x_list, f_list, weight=None, reference_point=None, nondom_archive=None, writer=writer, step=step)
        # if er.num_saved_policy < er.max_saved_policy:
        #     continue
        # policy.update_with_er(er, train_prob.reference_point, train_prob.nondom_archive, writer, step)
        policy.write_progress_to_tb(writer, step)

    return step, train_prob

def run(args):
    agent, policy, last_epoch, writer, checkpoint_path, test_env, sample_solutions = setup_r1_nes(args)
    num_nodes_list = [20,30]
    num_items_per_city_list = [1,3,5]
    config_list = [(num_nodes, num_items_per_city, idx) for num_nodes in num_nodes_list for num_items_per_city in num_items_per_city_list for idx in range(5)]
    num_configs = len(num_nodes_list)*len(num_items_per_city_list)
    step=1
    test_proc=None
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
        save_nes(policy, epoch, checkpoint_path)
        if test_proc is not None:
            test_proc.wait()
        # test_proc_cmd = "python validate.py --title "+ args.title + " --dataset-name "+ args.dataset_name + " --device cpu"
        test_proc_cmd = ["python",
                        "validate_r1nes.py",
                        "--title",
                        args.title,
                        "--dataset-name",
                        args.dataset_name,
                        "--device",
                        "cpu"]
        test_proc = subprocess.Popen(test_proc_cmd)
        # test_one_epoch(agent, policy, test_env, sample_solutions, writer, epoch)
        
    if test_proc is not None:
        test_proc.wait()

if __name__ == '__main__':
    args = prepare_args()
    # torch.set_num_threads(os.cpu_count())
    torch.set_num_threads(16)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)