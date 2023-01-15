import os
import random
import sys

import numpy as np
import torch

from arguments import get_parser
from setup import setup
from utils import write_test_progress
from utils import solve

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def test_one_epoch(agent, test_env, writer):
    agent.eval()
    tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve(agent, test_env)
    write_test_progress(tour_length, total_profit, total_cost, logprob, writer)    
        

def run(args):
    agent, agent_opt, last_epoch, writer, checkpoint_path, test_env = setup(args)
    test_one_epoch(agent, test_env, writer)

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)