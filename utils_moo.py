import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import get_parser
from arguments import get_parser
from ttp.ttp_env import TTPEnv
from policy.non_dominated_sorting import fast_non_dominated_sort
from utils import encode, solve_decode_only
from policy.normalization import normalize
from policy.hv import Hypervolume

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def decode_one_batch(agent, param_dict_list, train_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static):
    pop_size = len(param_dict_list)
    batch_size = train_env.batch_size
    travel_time_list = np.zeros((pop_size, batch_size), dtype=np.float32)
    total_profit_list = np.zeros((pop_size, batch_size), dtype=np.float32)
    logprobs_list = torch.zeros((pop_size, batch_size), dtype=torch.float32, device=agent.device)
    sum_entropies_list = torch.zeros((pop_size, batch_size), dtype=torch.float32, device=agent.device)
    for n, param_dict in enumerate(param_dict_list):
        solve_output = solve_decode_only(agent, train_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        travel_time_list[n,:] = tour_lengths
        total_profit_list[n,:] = total_profits
        logprobs_list[n,:] = logprobs
        sum_entropies_list[n, :] = sum_entropies
    inv_total_profit_list = -total_profit_list 
    f_list = np.concatenate((travel_time_list[:,:,np.newaxis],inv_total_profit_list[:,:,np.newaxis]), axis=-1)
    return logprobs_list, f_list,  sum_entropies_list


def solve_one_batch(agent, param_dict_list, batch):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp = batch
    env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp)
    # encode/embed first, it can be reused for same env/problem
    static_features = env.get_static_features()
    num_nodes, num_items, batch_size = env.num_nodes, env.num_items, env.batch_size
    encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output
    
    pop_size = len(param_dict_list)
    travel_time_list = np.zeros((pop_size, batch_size), dtype=np.float32)
    total_profit_list = np.zeros((pop_size, batch_size), dtype=np.float32)
        
    for n, param_dict in enumerate(tqdm(param_dict_list, desc="Solve Batch")):
        solve_output = solve_decode_only(agent, env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        travel_time_list[n,:] = tour_lengths
        total_profit_list[n,:] = total_profits
    inv_total_profit_list = -total_profit_list 
    
    f_list = np.concatenate((travel_time_list[:,:,np.newaxis],inv_total_profit_list[:,:,np.newaxis]), axis=-1)
    return f_list



@torch.no_grad()        
def validate_one_epoch(args, agent, policy, validator, validation_dataset, test_batch, test_sample_solutions, tb_writer, epoch):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, )
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=50, use_antithetic=False)
    f_list = []
    max_profits_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        batch_f_list = solve_one_batch(agent, param_dict_list, batch) 
        _, _, _, _, profits, _, _, _, _, _, _, _, _, _, _, _, _ = batch
        max_profits = profits.numpy().sum(axis=-1)
        f_list += [batch_f_list] 
        max_profits_list += [max_profits]
    f_list = np.concatenate(f_list,axis=1)
    if validator.initial_utopia_points is None:
        nadir_points = np.max(f_list, axis=0)
        utopia_points = np.min(f_list, axis=0)
        validator.set_initial_utopia_points(utopia_points)
        validator.set_initial_nadir_points(nadir_points)
        validator.compute_init_box_area()

    
    # if validator.extreme_utopia_points is None:
    #     max_profits_list = np.concatenate(max_profits_list)
    #     min_travel_distance = np.zeros_like(max_profits_list)
    #     extreme_utopia_points = np.concatenate([min_travel_distance[:,np.newaxis],-max_profits_list[:,np.newaxis]], axis=1)    
    #     validator.set_extreme_utopia_points(extreme_utopia_points)
        # nadir_points = np.max(f_list, axis=0)
        # validator.set_initial_nadir_points(nadir_points)
        # validator.compute_init_box_area()

    f_list = f_list.transpose((1,0,2))
    nadir_points = np.max(f_list, axis=1)
    utopia_points = np.min(f_list, axis=1)
    validator.insert_new_ref_points(nadir_points, utopia_points)

    nd_solutions_list = []
    for i in range(len(validation_dataset)):
        nondom_idx = fast_non_dominated_sort(f_list[i,:,:])[0]
        nd_solutions = f_list[i, nondom_idx, :]
        nd_solutions_list += [nd_solutions]
    validator.insert_new_nd_solutions(nd_solutions_list)
    validator.compute_new_hv(nd_solutions_list)
    validator.epoch +=1

    last_mean_running_igd = validator.get_last_mean_running_igd()
    if last_mean_running_igd is not None:
        tb_writer.add_scalar("Mean Running IGD", last_mean_running_igd, validator.epoch)
    last_mean_delta_nadir, last_mean_delta_utopia = validator.get_last_delta_refpoints()
    if last_mean_delta_nadir is not None:
        tb_writer.add_scalar("Mean Delta Nadir", last_mean_delta_nadir, validator.epoch)
        tb_writer.add_scalar("Mean Delta Utopia", last_mean_delta_utopia, validator.epoch)

    # Scatter plot with gradient colors
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=50, use_antithetic=False)
    test_f_list = solve_one_batch(agent, param_dict_list, test_batch)
    # Define the light and dark blue colors

    plt.figure()
    plt.scatter(test_sample_solutions[:,0], test_sample_solutions[:,1], c="red")
    plt.scatter(test_f_list[:,0,0], -test_f_list[:,0,1], c="blue")
    tb_writer.add_figure("Solutions "+args.dataset_name, plt.gcf(), epoch)
    tb_writer.add_scalar("Validation mean HV",validator.hv_list[-1,:].mean(), epoch)
    write_test_hv(tb_writer, test_f_list[:,0,:], epoch, test_sample_solutions)

    for i in range(len(validation_dataset)):
        plt.figure()
        plt.scatter(nd_solutions_list[i][:,0], nd_solutions_list[i][:,1], c="blue")
        plt.scatter(validator.nd_solutions_list[-1][i][:,0], validator.nd_solutions_list[-1][i][:,1], c="red", marker="1")
        tb_writer.add_figure("Solutions Validation-"+str(i), plt.gcf(), epoch)
        



def write_test_hv(writer, f_list, epoch, sample_solutions=None):
    # write the HV
    # get nadir and ideal point first
    all = np.concatenate([f_list, sample_solutions])
    ideal_point = np.min(all, axis=0)
    nadir_point = np.max(all, axis=0)
    _N = normalize(f_list, ideal_point, nadir_point)
    _hv = Hypervolume(np.array([1,1])).calc(_N)
    writer.add_scalar('Test HV', _hv, epoch)
    writer.flush()
