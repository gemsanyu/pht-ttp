import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from arguments import get_parser
from setup_phn import setup_phn
from utils import solve_decode_only, encode

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def test_one_epoch(agent, phn, test_env, x_file, y_file, time_file, n_solutions=200):
    agent.eval()
    phn.eval()
    
    encode_start = time.time()
    static_features, _, _ = test_env.begin()
    num_nodes, num_items, batch_size = test_env.num_nodes, test_env.num_items, test_env.batch_size
    static_embeddings = encode(agent, static_features, num_nodes, num_items, batch_size)
    encode_elapsed_time = (time.time()-encode_start)

    decode_start = time.time()
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    for ray in ray_list:
        param_dict = phn(ray.to(agent.device))
        solve_output = solve_decode_only(agent, test_env, static_embeddings, param_dict)
        tour_list, item_selection, tour_length, total_profit, total_costs, logprobs, sum_entropies = solve_output
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
        # print(tour_length+" "+total_profit+"\n")  
    decode_elapsed_time = (time.time()-decode_start)
    encode_elapsed_str = "{:.16f}".format(encode_elapsed_time)
    decode_elapsed_str = "{:.16f}".format(decode_elapsed_time)
    time_file.write(encode_elapsed_str+" "+decode_elapsed_str+"\n")

def run(args):
    agent, phn, phn_opt, last_epoch, writer, checkpoint_path, test_env, test_sample_solutions = setup_phn(args, load_best=True)
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    time_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".time")
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file, open(time_file_path.absolute(), "w") as time_file:
        test_one_epoch(agent, phn, test_env, x_file, y_file, time_file)

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    args = prepare_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)