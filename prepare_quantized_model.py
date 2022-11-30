import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

from arguments import get_parser
from setup import setup
from utils import solve
# from utils import solve_fast as solve

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
    tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve(agent, test_env)
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
    agent, _, _, _, checkpoint_path, _ = setup(args)
    quantized_path = checkpoint_path.parent / (checkpoint_path.name + "quantized")
    quantized_agent = torch.quantization.quantize_dynamic(
        agent, qconfig_spec={torch.nn.Linear, torch.nn.InstanceNorm1d}, dtype=torch.qint8
    )
    qa_sd = quantized_agent.state_dict()
    gae_query_key_list = []
    for i in range(3):  
        gae_query_key_list += ["gae.layers."+str(i)+".0.module.W_query"]
        gae_query_key_list += ["gae.layers."+str(i)+".0.module.W_key"]
        gae_query_key_list += ["gae.layers."+str(i)+".0.module.W_val"]
        gae_query_key_list += ["gae.layers."+str(i)+".0.module.W_out"]

    for k, v in qa_sd.items():
        print(k, v.dtype)

    for k in gae_query_key_list:    
        qa_sd[k] = torch.quantize_per_tensor(qa_sd[k], 1e-4, 2, torch.qint8)
        print(k, qa_sd[k].dtype)

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(os.cpu_count()-4)
    # torch.set_num_threads(16)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)