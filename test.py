import os
import pathlib
import random
import sys

import numpy as np
import torch

from arguments import get_parser
from setup import setup_r1_nes
from utils import solve

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    return args

@torch.no_grad()
def test_one_epoch(agent, policy, test_env, x_file, y_file, pop_size=200):
    agent.eval()
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    for n, param_dict in enumerate(param_dict_list):
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, test_env, param_dict)
        node_order_str = ""
        for i in tour_list[0]:
            node_order_str+= str(i.item()) + " "
        x_file.write(node_order_str+"\n")
        item_selection_str = ""
        for i in item_selection[0]:
            item_selection_str += (str(int(i.item()))) + " "
        x_file.write(item_selection_str+"\n")

        tour_length = "{:.16f}".format(tour_lengths[0].item())
        total_profit = "{:.16f}".format(total_profits[0].item())
        y_file.write(tour_length+" "+total_profit+"\n")
        print(tour_length+" "+total_profit+"\n")

def test(args):
    agent, policy, last_epoch, writer, checkpoint_path, test_env, sample_solutions = setup_r1_nes(args)
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file:
        test_one_epoch(agent, policy, test_env, x_file, y_file)


if __name__=='__main__':
    args = prepare_args()
    #torch.set_num_threads(os.cpu_count()-4)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    test(args)
