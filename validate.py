from cgi import test
import os
import random
import sys

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from arguments import get_parser
from setup import setup_phn
from utils import write_test_phn_progress
from utils import solve_decode_only

CPU_DEVICE = torch.device("cpu")


def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def test_one_epoch(agent, phn, test_env, test_sample_solutions, writer, epoch, n_solutions=20):
    agent.eval()
    phn.eval()
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    solution_list = []
    # across rays, the static embeddings are the same, so reuse
    static_features = test_env.get_static_features()
    static_features = torch.from_numpy(static_features).to(agent.device)
    static_embeddings, graph_embeddings = agent.gae(static_features)
    for ray in tqdm(ray_list, desc="Testing"):
        param_dict = phn(ray.to(agent.device))
        tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve_decode_only(agent, test_env, static_embeddings, graph_embeddings, param_dict)
        solution_list += [torch.stack([tour_length, total_profit], dim=1)]
    solution_list = torch.cat(solution_list)
    ray_list = torch.cat(ray_list, dim=0)
    write_test_phn_progress(writer, solution_list, ray_list, epoch, test_sample_solutions)

def run(args):
    agent, phn, _, last_epoch, writer, _, test_env, test_sample_solutions = setup_phn(args)
    test_one_epoch(agent, phn, test_env, test_sample_solutions, writer, last_epoch)

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    args = prepare_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)