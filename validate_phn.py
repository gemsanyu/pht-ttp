import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from arguments import get_parser
from policy.hv import Hypervolume
from setup import setup_phn
from utils import write_test_phn_progress
from utils import solve_decode_only, encode
from validator import load_validator, save_validator

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def validate(agent, phn, validation_env_list, n_solutions=50):
    # sample once for every env?
    agent.eval()
    num_env = len(validation_env_list)
    batch_size = validation_env_list[0].batch_size
    
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    param_dict_list  = [phn(ray.to(agent.device)) for ray in ray_list]
    solution_list = torch.zeros((num_env, batch_size, n_solutions, 2), dtype=torch.float32)    

    for env_idx, env in enumerate(tqdm(validation_env_list)):
        static_features = env.get_static_features()
        num_nodes, num_items, batch_size = env.num_nodes, env.num_items, env.batch_size
        encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
        static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output
    
        travel_time_list = torch.zeros((batch_size, n_solutions, 1), dtype=torch.float32)
        total_profit_list = torch.zeros((batch_size, n_solutions, 1), dtype=torch.float32)
    
        for n, param_dict in enumerate(param_dict_list):
            solve_output = solve_decode_only(agent, env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
            tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
            travel_time_list[:, n, 0] = tour_lengths
            total_profit_list[:, n, 0] = total_profits
        env_solution_list = torch.cat([travel_time_list, total_profit_list], dim=-1)
        solution_list[env_idx] = env_solution_list
    solution_list[:,:,:,1] = -solution_list[:,:,:,1]
    return solution_list    

def compute_hv(solution_list, nadir_points, utopia_points):
    # print(solution_list.shape)
    num_env, batch_size, _, _ = solution_list.shape
    new_nadir_points = torch.zeros_like(nadir_points)
    new_utopia_points = torch.zeros_like(utopia_points)
    new_hv = torch.zeros((num_env, batch_size, 1))
    reference_point = np.array([1.1,1.1])
    hv_getter = Hypervolume(reference_point)
    for env_idx in range(num_env):
        for batch_idx in range(batch_size):
            env_solutions = solution_list[env_idx, batch_idx]
            new_nadir_point, _ = torch.max(env_solutions, dim=0)
            new_utopia_point, _ = torch.min(env_solutions, dim=0)
            old_nadir_point = nadir_points[env_idx, batch_idx]
            old_utopia_point = utopia_points[env_idx, batch_idx]
            new_nadir_point = torch.maximum(new_nadir_point, old_nadir_point)
            new_utopia_point = torch.minimum(new_utopia_point, old_utopia_point)
            new_nadir_points[env_idx, batch_idx] = new_nadir_point
            new_utopia_points[env_idx, batch_idx] = new_utopia_point
            new_nadir_point, new_utopia_point = new_nadir_point.unsqueeze(0), new_utopia_point.unsqueeze(0)
            _N = (env_solutions-new_utopia_point)/(new_nadir_point-new_utopia_point)
            hv = hv_getter.calc(_N.numpy())
            new_hv[env_idx, batch_idx] = hv
    
    return new_hv, new_nadir_points, new_utopia_points
    

@torch.no_grad()
def test_one_epoch(agent, phn, test_env, n_solutions=30):
    agent.eval()
    phn.eval()
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    solution_list = []
    # across rays, the static embeddings are the same, so reuse
    static_features = test_env.get_static_features()
    num_nodes, num_items, batch_size = test_env.num_nodes, test_env.num_items, test_env.batch_size
    encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output

    for ray in tqdm(ray_list, desc="Testing"):
        param_dict = phn(ray.to(agent.device))
        solve_output = solve_decode_only(agent, test_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve_output
        solution_list += [torch.stack([tour_length, total_profit], dim=1)]
    solution_list = torch.cat(solution_list)
    ray_list = torch.cat(ray_list, dim=0)
    return solution_list
    
def run(args):
    agent, phn, _, last_epoch, writer, test_env, test_sample_solutions = setup_phn(args)
    vd = load_validator(args.title)
    solution_list = validate(agent, phn, vd.validation_env_list)
    new_hv, new_nadir_points, new_utopia_points = compute_hv(solution_list, vd.nadir_points, vd.utopia_points) 
    vd.update_ref_points(new_nadir_points, new_utopia_points)
    vd.insert_hv_history(new_hv)
    vd.epoch += 1
    test_solution_list = test_one_epoch(agent, phn, test_env)
    write_test_phn_progress(writer, test_solution_list, vd.epoch, test_sample_solutions)
    writer.add_scalar("Validation Mean HV", vd.last_hv)
    save_validator(vd, args.title)
    
if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    args = prepare_args()
    torch.set_num_threads(16)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)