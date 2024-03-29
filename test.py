import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

from agent.agent import Agent
from agent.encoder import Encoder
from setup_r1nes import setup_r1_nes
from policy.r1_nes import R1_NES
from utils import solve_decode_only, encode, prepare_args, CPU_DEVICE
from ttp.ttp_env import TTPEnv


@torch.no_grad()
def test_one_epoch(agent:Agent, encoder:Encoder, policy:R1_NES, test_batch, x_file, y_file, time_file, pop_size=15):
    agent.eval()
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    
    encode_start = time.time()
    static_features, _, _ = test_env.begin()
    num_nodes, num_items, batch_size = test_env.num_nodes, test_env.num_items, test_env.batch_size
    static_embeddings = encode(encoder, static_features, num_nodes, num_items, batch_size)
    encode_elapsed_time = (time.time()-encode_start)
    
    decode_start = time.time()
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False, device=encoder.device)   
    for pi, param_dict in enumerate(param_dict_list):
        print(pi)
        solve_output = solve_decode_only(agent, encoder, test_env, static_embeddings, param_dict)
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
    agent, encoder, policy, training_nondom_list, validation_nondom_list, best_f_list, last_epoch, writer, checkpoint_path, test_batch, sample_solutions = setup_r1_nes(args, load_best=True)
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    time_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".time")
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file, open(time_file_path.absolute(), "w") as time_file:
        test_one_epoch(agent, encoder, policy, test_batch, x_file, y_file, time_file)

if __name__ == '__main__':
    args = prepare_args()
    # torch.set_num_threads(2)
    # torch.set_num_threads(os.cpu_count()-4)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)