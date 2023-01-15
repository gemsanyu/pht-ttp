import math
import subprocess
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from agent.agent import Agent
from arguments import get_parser
from setup_r1nes import setup_r1_nes
from ttp.ttp_dataset import read_prob, TTPDataset, combine_batch_list
from ttp.ttp import TTP
from ttp.ttp_env import TTPEnv
from policy.r1_nes import R1_NES
from policy.utils import get_score_hv_contributions
from utils import save_nes, solve_decode_only
from utils import encode
from validator import load_validator

CPU_DEVICE = torch.device("cpu")
MAX_PATIENCE = 50

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args


def solve_one_batch(agent, param_dict_list, batch):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
    train_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    # encode/embed first, it can be reused for same env/problem
    static_features = train_env.get_static_features()
    num_nodes, num_items, batch_size = train_env.num_nodes, train_env.num_items, train_env.batch_size
    encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output
    
    pop_size = len(param_dict_list)
    travel_time_list = torch.zeros((pop_size, batch_size), dtype=torch.float32)
    total_profit_list = torch.zeros((pop_size, batch_size), dtype=torch.float32)
        
    for n, param_dict in enumerate(tqdm(param_dict_list, desc="Solve Batch")):
        solve_output = solve_decode_only(agent, train_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        travel_time_list[n,:] = tour_lengths
        total_profit_list[n,:] = total_profits
    inv_total_profit_list = -total_profit_list 
    f_list = torch.cat((travel_time_list.unsqueeze(2),inv_total_profit_list.unsqueeze(2)), dim=-1)
    return f_list

@torch.no_grad()
def train_one_generation(agent:Agent, policy: R1_NES, batch_list, pop_size=10):
    agent.eval()
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    f_list = []

    for batch in tqdm(batch_list, desc="Generation"):
        batch_f_list = solve_one_batch(agent, param_dict_list, batch)    
        f_list += [batch_f_list]
    f_list = torch.cat(f_list, dim=1)
    _, total_batch_size, _ = f_list.shape
    score = []
    for batch_idx in range(total_batch_size):
        batch_score = get_score_hv_contributions(f_list[:,batch_idx,:], policy.negative_hv)    
        score += [batch_score]
    score = torch.cat(score, dim=-1)
    score = score.mean(dim=1, keepdim=True)
    x_list = sample_list - policy.mu
    w_list = x_list/math.exp(policy.ld)
    policy.update(w_list, x_list, score)

def run(args):
    agent, policy, last_epoch, writer, checkpoint_path, test_env, sample_solutions = setup_r1_nes(args)
    num_nodes_list = [20,30]
    num_items_per_city_list = [1,3,5]
    ic_list = [0,1,2]
    num_config = len(num_nodes_list)*len(num_items_per_city_list)*len(ic_list)
    batch_size_per_config = int(args.batch_size/num_config)
    config_list = [(num_nodes, num_items_per_city, ic) for num_nodes in num_nodes_list for num_items_per_city in num_items_per_city_list for ic in ic_list]
    datasets = [TTPDataset(64, config[0], config[1], config[2]) for config in config_list]
    step=1  
    vd_proc:subprocess.Popen=None
    epoch = last_epoch
    early_stop = 0
    while epoch < args.max_epoch:
        dl_iter_list = [iter(DataLoader(dataset, batch_size=batch_size_per_config, shuffle=True)) for dataset in datasets]
        for i in range(4):
            if early_stop == MAX_PATIENCE:
                break
            batch_list = [next(dl_iter) for dl_iter in dl_iter_list]
            batch_list = [combine_batch_list([batch_list[i],batch_list[i+1],batch_list[i+2]]) for i in range(0,18,3)]# hasil kombinasi yg jumlah elemen sama
            train_one_generation(agent, policy, batch_list, pop_size=policy.pop_size)
            policy.write_progress_to_tb(writer, step)
            # Validate dulu baru save jika masih ada progress?
            if vd_proc is not None:
                vd_proc.wait()
            vd = load_validator(args.title)
            if vd.is_improving:
                early_stop = 0
                save_nes(policy, epoch, args.title, best=True)
            else:   
                early_stop += 1
            save_nes(policy, epoch, args.title)
            vd_proc_cmd = ["python",
                        "validate_r1nes.py",
                        "--title",
                        args.title,
                        "--dataset-name",
                        args.dataset_name,
                        "--device",
                        "cpu"]
            vd_proc = subprocess.Popen(vd_proc_cmd)
            epoch += 1
        if early_stop == MAX_PATIENCE:
            break
    vd_proc.wait()

if __name__ == '__main__':
    args = prepare_args()
    # torch.set_num_threads(os.cpu_count())
    torch.set_num_threads(16)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)