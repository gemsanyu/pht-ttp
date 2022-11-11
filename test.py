import os
import pathlib
import random
import sys

import numpy as np
import torch
import torch.quantization

from arguments import get_parser
from setup import setup
# from utils import solve
from utils import solve_fast as solve

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def test_one_epoch(agent, test_env, x_file, y_file):
    agent.eval()
    tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve(agent, test_env, k=100)
    node_order_str = ""
    for i in tour_list[0]:
        node_order_str+= str(i.item()) + " "
    x_file.write(node_order_str+"\n")
    item_selection_str = ""
    for i in item_selection[0]:
        item_selection_str += (str(int(i.item()))) + " "
    x_file.write(item_selection_str+"\n")

    tour_length = "{:.16f}".format(tour_length[0].item())
    total_profit = "{:.16f}".format(total_profit[0].item())
    y_file.write(tour_length+" "+total_profit+"\n")
    print(tour_length+" "+total_profit+"\n")    


def run(args):
    agent, agent_opt, last_epoch, writer, checkpoint_path, test_env = setup(args)
    results_dir = summary_dir = pathlib.Path(".")/"results"
    # agent = torch.quantization.quantize_dynamic(agent, {torch.nn.Linear}, dtype=torch.qint8)
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file:
        test_one_epoch(agent, test_env, x_file, y_file)

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(os.cpu_count())
    # torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)