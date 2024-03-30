import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from setup_r1nes_mp import setup_r1_nes
from utils import solve_decode_only, encode
from utils import prepare_args, CPU_DEVICE
from ttp.ttp_env import TTPEnv

@torch.no_grad()
def decode_mp(pi, param_dict, test_batch, device_str, encode_result):
    print(pi)
    device = torch.device(device_str)
    param_dict["po_weight"] = param_dict["po_weight"].to(device)
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_result
    static_embeddings = static_embeddings.to(device)
    fixed_context = fixed_context.to(device)
    glimpse_K_static = glimpse_K_static.to(device)
    glimpse_V_static = glimpse_V_static.to(device)
    logits_K_static = logits_K_static.to(device)
    agent = torch.jit.load("traced_agent", map_location=device)
    solve_output = solve_decode_only(agent, device, test_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
    tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve_output
    tour_length = "{:.16f}".format(tour_length[0].item())
    total_profit = "{:.16f}".format(total_profit[0].item())
    return (tour_length, total_profit)

@torch.no_grad()
def test_mp(encoder, policy, test_batch, x_file, y_file, pop_size=200):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    
    #get static embeddings first, it can be widely reused
    static_features = test_env.get_static_features()
    num_nodes, num_items, batch_size = test_env.num_nodes, test_env.num_items, test_env.batch_size
    encode_output = encode(encoder, static_features, num_nodes, num_items, batch_size)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output
    static_embeddings = static_embeddings.to(CPU_DEVICE)
    fixed_context = fixed_context.to(CPU_DEVICE)
    glimpse_K_static = glimpse_K_static.to(CPU_DEVICE)
    glimpse_V_static = glimpse_V_static.to(CPU_DEVICE)
    logits_K_static = logits_K_static.to(CPU_DEVICE)
    encode_output = (static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
    
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    for i in range(len(param_dict_list)):
        param_dict_list[i]["po_weight"] = param_dict_list[i]["po_weight"].to(CPU_DEVICE)
    input_list = [(pi, param_dict, test_batch, "cuda", encode_output) for pi,param_dict in enumerate(param_dict_list)]
    
    num_cpu = 12
    with mp.Pool(num_cpu) as pool:
        for result in pool.starmap(decode_mp, input_list):
            tour_length, total_profit = result
            y_file.write(tour_length+" "+total_profit+"\n")
    

def run(args):
    encoder, policy, test_batch = setup_r1_nes(args, load_best=False)
    # agent.gae = agent.gae.cpu()
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file:
        test_mp(encoder, policy, test_batch, x_file, y_file)

if __name__ == '__main__':
    args = prepare_args()
    mp.set_start_method("spawn")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)
