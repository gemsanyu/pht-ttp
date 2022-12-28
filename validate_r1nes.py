import math
from multiprocessing import Pool
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from agent.agent import Agent
from arguments import get_parser
from setup import setup_r1_nes
from utils import write_test_phn_progress, solve_decode_only

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def test_one_epoch(agent:Agent, policy, test_env, sample_solutions, writer, epoch, pop_size=100):
    agent.eval()
    static_features = test_env.get_static_features()
    static_features = torch.from_numpy(static_features).to(agent.device)
    item_init_embed = agent.item_init_embedder(static_features[:, :test_env.num_items, :])
    depot_init_embed = agent.depot_init_embed.expand(size=(test_env.batch_size,1,-1))
    node_init_embed = agent.node_init_embed.expand(size=(test_env.batch_size,test_env.num_nodes-1,-1))
    init_embed = torch.cat([item_init_embed, depot_init_embed, node_init_embed], dim=1)
    static_embeddings, graph_embeddings = agent.gae(init_embed)
    fixed_context = agent.project_fixed_context(graph_embeddings)
    glimpse_K_static, glimpse_V_static, logits_K_static = agent.project_embeddings(static_embeddings).chunk(3, dim=-1)
    glimpse_K_static = agent._make_heads(glimpse_K_static)
    glimpse_V_static = agent._make_heads(glimpse_V_static)

    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    solution_list = []
    for n, param_dict in enumerate(tqdm(param_dict_list)):
        solve_output = solve_decode_only(agent, test_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        solution_list += [torch.stack([tour_lengths, total_profits], dim=1)]
    solution_list = torch.cat(solution_list)
    write_test_phn_progress(writer, solution_list, epoch, sample_solutions)

def run(args):
    agent, policy, last_epoch, writer, checkpoint_path, test_env, sample_solutions = setup_r1_nes(args)
    test_one_epoch(agent, policy, test_env, sample_solutions, writer, last_epoch)

if __name__ == '__main__':
    args = prepare_args()
    # torch.set_num_threads(os.cpu_count())
    torch.set_num_threads(12)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)