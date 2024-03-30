import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from agent.encoder import load_encoder
from arguments import get_parser
from setup_phn_mp import setup_phn_mp
from utils import solve_decode_only, encode
from utils import prepare_args, CPU_DEVICE
from ttp.ttp_env import TTPEnv

@torch.no_grad()
def decode_mp(pi, test_batch, param_dict, static_embeddings, device_str="cuda"):
    print(pi)
    device = torch.device(device_str)
    agent = torch.jit.load("traced_agent-nrw.pt", map_location=device)
    encoder = load_encoder(device_str)
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    static_embeddings = static_embeddings.to(device)
    solve_output = solve_decode_only(agent, encoder, test_env, static_embeddings, param_dict)
    tour_list, item_selection, tour_length, total_profit, total_costs, logprobs, sum_entropies = solve_output
    tour_length = "{:.16f}".format(tour_length[0].item())
    total_profit = "{:.16f}".format(total_profit[0].item())
    result = (tour_length, total_profit)
    return result

@torch.no_grad()
def test_mp(encoder, phn, test_batch, x_file, y_file, n_solutions=200):
    phn.eval()
    
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    
    static_features, _, _ = test_env.begin()
    num_nodes, num_items, batch_size = test_env.num_nodes, test_env.num_items, test_env.batch_size
    static_embeddings = encode(encoder, static_features, num_nodes, num_items, batch_size)
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    param_dict_list = [phn(ray.to(phn.device)) for ray in ray_list]
    static_embeddings = static_embeddings.to(CPU_DEVICE)
    input_list = [(pi, test_batch, param_dict, static_embeddings, "cuda") for pi, param_dict in enumerate(param_dict_list)]
    num_proc = 12
    with mp.Pool(num_proc) as pool:
        for result in pool.starmap(decode_mp, input_list):
            tour_length, total_profit = result
            y_file.write(tour_length+" "+total_profit+"\n")


def run(args):
    encoder, phn, test_batch = setup_phn_mp(args, load_best=False)
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file:
        test_mp(encoder, phn, test_batch, x_file, y_file)

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    mp.set_start_method("spawn")
    args = prepare_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)