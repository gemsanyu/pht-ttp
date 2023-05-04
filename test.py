import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


from arguments import get_parser
from setup import setup
from utils import solve_decode_only, encode
from agent.agent import Agent
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv
# from utils import solve_fast as solve

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def test_one_epoch(agent, test_env, x_file, y_file):
    agent.eval()
    static_features, _, _, _ = test_env.begin()
    num_nodes, num_items, batch_size = test_env.num_nodes, test_env.num_items, test_env.batch_size
    encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output
    tour_list, item_selection, tour_length, total_profit, travel_cost, total_cost, logprob, sum_entropies = solve_decode_only(agent, test_env, static_embeddings, fixed_context,glimpse_K_static, glimpse_V_static, logits_K_static)
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

def load_agent_checkpoint(agent, title, weight_idx, total_weight, device=CPU_DEVICE):
    agent_title = title + str(weight_idx) + "_" + str(total_weight)
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/agent_title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")
    checkpoint = torch.load(checkpoint_path.absolute(), map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    return agent

def run(args):
    agent = Agent(n_heads=8,
                 num_static_features=3,
                 num_dynamic_features=4,
                 n_gae_layers=3,
                 embed_dim=128,
                 gae_ff_hidden=128,
                 tanh_clip=10,
                 device=args.device)    
    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    test_batch = next(iter(test_dataloader))
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)

    elapsed_time = 0
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    for weight_idx in range(1,args.total_weight+1):
        agent = load_agent_checkpoint(agent, args.title, weight_idx, args.total_weight, args.device)        
        x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
        y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
        start = time.time()
        with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file:
            test_one_epoch(agent, test_env, x_file, y_file)
        elapsed_time += (time.time()-start)
    time_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".time")
    with open(time_file_path.absolute(), "a+") as time_file:
        elapsed_time_str = "{:.16f}".format(elapsed_time)
        time_file.write(elapsed_time_str+"\n")

if __name__ == '__main__':
    args = prepare_args()
    # torch.set_num_threads(os.cpu_count()-4)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)