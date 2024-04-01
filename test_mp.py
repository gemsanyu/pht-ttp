import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from agent.agent import Agent
from agent.encoder import Encoder, load_encoder
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv
from utils import solve, prepare_args, CPU_DEVICE

@torch.no_grad()
def decode_mp(test_batch, title, weight_idx, total_weight):
    print(weight_idx)
    device = torch.device("cuda")
    agent, encoder = load_agent_and_encoder(title, weight_idx, total_weight, device)
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)

    tour_list, item_selection, tour_length, total_profit, travel_cost, total_cost, logprob, sum_entropies = solve(agent, encoder, test_env)
    tour_length = "{:.16f}".format(tour_length[0].item())
    total_profit = "{:.16f}".format(total_profit[0].item())
    return (tour_length, total_profit)

def load_agent_and_encoder(title, weight_idx, total_weight, device=CPU_DEVICE):
    traced_name = title+"_"+str(weight_idx)+"_"+str(total_weight)+"traced_agent.pt"
    traced_dir = pathlib.Path()/"traced_agent"
    traced_path = (traced_dir/traced_name).absolute()
    agent = torch.jit.load(traced_path, device)
    encoder = load_encoder(device, title, weight_idx, total_weight)
    return agent, encoder

def run(args):
    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    test_batch = next(iter(test_dataloader))
    
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    weight_idx_list = list(range(1,args.total_weight+1))
    config_list = [(test_batch, args.title, weight_idx, args.total_weight) for weight_idx in weight_idx_list]
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    num_cpu = 12      
    with mp.Pool(num_cpu) as pool, open(y_file_path.absolute(), "a+") as y_file:
        for result in pool.starmap(decode_mp, config_list):
            tour_length, total_profit = result
            y_file.write(tour_length+" "+total_profit+"\n")
    

if __name__ == '__main__':
    args = prepare_args()
    mp.set_start_method("spawn")
    # torch.set_num_threads(os.cpu_count()-4)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)

