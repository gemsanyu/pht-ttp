import pathlib
import random

import numpy as np
import torch
import torch.multiprocessing as mp

from agent.agent import Agent
from agent.encoder import Encoder, load_encoder
from setup_r1nes_mp import setup_r1nes_mp
from policy.r1_nes import R1_NES
from utils import solve_decode_only, encode, prepare_args, CPU_DEVICE
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
def test_mp(encoder:Encoder, policy:R1_NES, test_batch, x_file, y_file, pop_size=200):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    
    static_features, _, _ = test_env.begin()
    num_nodes, num_items, batch_size = test_env.num_nodes, test_env.num_items, test_env.batch_size
    static_embeddings = encode(encoder, static_features, num_nodes, num_items, batch_size)
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False, device=encoder.device)   
    static_embeddings = static_embeddings.to(CPU_DEVICE)
    input_list = [(pi, test_batch, param_dict, static_embeddings, "cuda") for pi, param_dict in enumerate(param_dict_list)]
    num_proc = 5
    with mp.Pool(num_proc) as pool:
        for result in pool.starmap(decode_mp, input_list):
            tour_length, total_profit = result
            y_file.write(tour_length+" "+total_profit+"\n")

def run(args):
    encoder, policy, test_batch = setup_r1nes_mp(args, load_best=True)
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file:
        test_mp(encoder, policy, test_batch, x_file, y_file)

if __name__ == '__main__':
    args = prepare_args()
    # torch.set_num_threads(2)
    # torch.set_num_threads(os.cpu_count()-4)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)