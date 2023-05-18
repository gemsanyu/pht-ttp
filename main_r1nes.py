import math
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ranksums

from agent.agent import Agent
from setup_r1nes import setup_r1_nes
from ttp.ttp_env import TTPEnv
from ttp.ttp_dataset import get_dataset_list
from policy.r1_nes import R1_NES
from policy.hv import Hypervolume
from policy.utils import get_score_hv_contributions, fast_non_dominated_sort
from utils import solve_decode_only, encode, save_nes
from utils import prepare_args, write_test_hv

def solve_one_batch(agent, param_dict_list, batch, index_list, training_nondom_list, negative_hv, is_validate=False):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
    train_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    # encode/embed first, it can be reused for same env/problem
    num_nodes, num_items, batch_size = train_env.num_nodes, train_env.num_items, train_env.batch_size
    static_features = train_env.get_static_features()    
    num_nodes, num_items, batch_size = train_env.num_nodes, train_env.num_items, train_env.batch_size
    encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output
    pop_size = len(param_dict_list)
    travel_time_list = np.zeros((pop_size, batch_size), dtype=np.float32)
    total_profit_list = np.zeros((pop_size, batch_size), dtype=np.float32)
    for n, param_dict in enumerate(param_dict_list):
        solve_output = solve_decode_only(agent, 
                                         train_env, 
                                         static_embeddings, 
                                         fixed_context,
                                         glimpse_K_static, 
                                         glimpse_V_static, 
                                         logits_K_static, 
                                         param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        travel_time_list[n,:] = tour_lengths
        total_profit_list[n,:] = total_profits
    inv_total_profit_list = -total_profit_list 
    f_list = np.concatenate((travel_time_list[:,:,np.newaxis],inv_total_profit_list[:,:,np.newaxis]), axis=-1)
    if is_validate:
        return f_list, None, None
    score = []
    _, num_instances, _ = f_list.shape
    for i in range(num_instances):
        old_nondom_f = training_nondom_list[index_list[i]]
        f_list_ = f_list[:,i,:]
        combined_f = f_list_
        if old_nondom_f is not None:
            combined_f = np.concatenate([combined_f, old_nondom_f])
        nondom_f_idx = fast_non_dominated_sort(combined_f)[0]
        nondom_f = combined_f[nondom_f_idx, :]
        training_nondom_list[index_list[i]] = nondom_f
        max_f = np.max(nondom_f, axis=0, keepdims=True)
        min_f = np.min(nondom_f, axis=0, keepdims=True)
        diff = max_f-min_f
        diff[diff==0]=1
        norm_f_list = (f_list_-min_f)/diff
        batch_score = get_score_hv_contributions(norm_f_list, negative_hv)    
        score += [batch_score]
    score = np.concatenate(score, axis=-1)
    return f_list, training_nondom_list, score

@torch.no_grad()
def train_one_generation(args, agent:Agent, policy:R1_NES, training_nondom_list, writer, training_dataset_list, pop_size, epoch):
    agent.eval()
    batch_size_per_dataset = int(args.batch_size/len(training_dataset_list))
    training_dataloader_list = [enumerate(DataLoader(train_dataset, batch_size=batch_size_per_dataset, shuffle=True, pin_memory=True)) for train_dataset in training_dataset_list]
    if training_nondom_list is None:
        training_nondom_list = [[None for _ in range(len(training_dataset_list[i]))] for i in range(len(training_dataloader_list))]
    is_done=False
    while not is_done:
        score_list = []
        param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
        for i, dl_it in tqdm(enumerate(training_dataloader_list), desc="Training"):
            try:
                batch_idx, batch = next(dl_it)
                index_list, batch = batch
                batch_f_list, training_nondom_list[i], score = solve_one_batch(agent, param_dict_list, batch, index_list, training_nondom_list[i], policy.negative_hv)    
                score_list += [score]
            except StopIteration:
                is_done=True
                break 
        if is_done:
            break
        score_list = np.concatenate(score_list, axis=-1)
        score_list = torch.from_numpy(score_list)
        score = score_list.mean(dim=-1, keepdim=True)
        x_list = sample_list - policy.mu
        w_list = x_list/math.exp(policy.ld)
        policy.update(w_list, x_list, score)
    return training_nondom_list

@torch.no_grad()
def validate_one_epoch(args, agent, policy, validation_nondom_list, best_f_list, validation_dataset_list, writer, test_batch, test_sample_solutions, epoch):
    agent.eval()
    batch_size_per_dataset = int(args.batch_size/len(validation_dataset_list))
    validation_dataloader_list = [enumerate(DataLoader(validation_dataset, batch_size=batch_size_per_dataset, shuffle=False, pin_memory=True)) for validation_dataset in validation_dataset_list]
    
    #evaluate agent
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=policy.pop_size, use_antithetic=False)
    f_list = []
    is_done=False
    while not is_done:
        for dl_it in tqdm(validation_dataloader_list, desc="Validation"):
            try:
                batch_idx, batch = next(dl_it)
                index_list, batch = batch
                batch_f_list, _, _ = solve_one_batch(agent, param_dict_list, batch, None, None, None, True)
                f_list += [batch_f_list] 
            except StopIteration:
                is_done=True
                break
    f_list = np.concatenate(f_list,axis=1)
    if best_f_list is None:
        best_f_list = f_list
    f_root = "f_files"
    f_dir = pathlib.Path(".")/f_root
    model_f_dir = f_dir/args.title
    model_f_dir.mkdir(parents=True, exist_ok=True)
    f_path = model_f_dir/(args.title+"_"+str(epoch)+".pt")
    np.save(f_path.absolute(), f_list)

    # now compare the agent's solutions hv with the critics
    # use wilcoxon signed rank
    _, num_validation_instances, _ = f_list.shape
    if validation_nondom_list is None:
        validation_nondom_list = [None for _ in range(num_validation_instances)]
    
    hv_list = []
    best_hv_list = []
    for i in range(num_validation_instances):
        agent_f = f_list[:,i,:]
        best_f = best_f_list[:,i,:]
        old_nondom_f = validation_nondom_list[i]
        combined_f = np.concatenate([agent_f, best_f], axis=0)
        if old_nondom_f is not None:
            combined_f = np.concatenate([combined_f, old_nondom_f], axis=0)
        nondom_f_idx = fast_non_dominated_sort(combined_f)[0]
        nondom_f = combined_f[nondom_f_idx,:]
        validation_nondom_list[i] = nondom_f
        utopia_points = np.min(nondom_f, axis=0)
        nadir_points = np.max(nondom_f, axis=0)
        diff = nadir_points-utopia_points
        diff[diff==0] = 1
        norm_agent_f = (agent_f-utopia_points)/diff
        norm_best_f= (best_f-utopia_points)/diff
        agent_hv = Hypervolume(np.array([1.1,1.1])).calc(norm_agent_f)
        best_hv = Hypervolume(np.array([1.1,1.1])).calc(norm_best_f)
        hv_list += [agent_hv]
        best_hv_list += [best_hv]
    hv_list = np.asanyarray(hv_list)
    best_hv_list = np.asanyarray(best_hv_list)
    is_improving=False
    try:
        res = ranksums(hv_list, best_hv_list, alternative="greater")
        is_improving = res.pvalue < 0.05
    except ValueError:
        is_improving = False
    if is_improving:
        best_f_list = f_list
    # writer.add_scalar("Mean Validation HV",hv_list.mean(),epoch)
    # writer.add_scalar("Std Validation HV",hv_list.std(),epoch)
    # writer.add_scalar("Median Validation HV",np.median(hv_list),epoch)
    is_improving_val = 1 if is_improving else 0
    writer.add_scalar("is improving?", is_improving_val, epoch)
    
    # Scatter plot with gradient colors
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=200, use_antithetic=False)
    test_f_list,_,_ = solve_one_batch(agent, param_dict_list, test_batch, None,  None, None, True)
    
    plt.figure()
    plt.scatter(test_sample_solutions[:,0], -test_sample_solutions[:,1], c="red")
    nondom_idx = fast_non_dominated_sort(test_f_list[:,0,:])[0]
    nondom_test_f = test_f_list[nondom_idx,:,:]
    plt.scatter(nondom_test_f[:,0,0], nondom_test_f[:,0,1], c="blue")
    writer.add_figure("Solutions "+args.dataset_name, plt.gcf(), epoch)
    write_test_hv(writer, test_f_list[:,0,:], epoch, test_sample_solutions)
    return is_improving, validation_nondom_list, best_f_list

def run(args):
    agent, policy, training_nondom_list, validation_nondom_list, best_f_list, last_epoch, writer, checkpoint_path, test_batch, sample_solutions = setup_r1_nes(args)
    nn_list = [10,20,30]
    nipc_list = [1,3,5]
    len_types = len(nn_list)*len(nipc_list)
    train_num_samples_per_dataset = int(args.num_training_samples/len_types)
    validation_num_samples_per_dataset = int(args.num_validation_samples/len_types)
    training_dataset_list = get_dataset_list(train_num_samples_per_dataset, nn_list, nipc_list, mode="training")
    validation_dataset_list = get_dataset_list(validation_num_samples_per_dataset, nn_list, nipc_list, mode="validation")

    patience = 30
    not_improving_count = 0
    #get initial validation_nondom_list
    is_improving, validation_nondom_list, best_f_list = validate_one_epoch(args, agent, policy, validation_nondom_list, best_f_list, validation_dataset_list, writer, test_batch, sample_solutions, -1)  
    epoch = last_epoch
    for epoch in range(last_epoch, args.max_epoch):
        training_nondom_list = train_one_generation(args, agent, policy, training_nondom_list, writer, training_dataset_list, policy.pop_size, epoch)
        policy.write_progress_to_tb(writer, epoch)
        is_improving, validation_nondom_list, best_f_list = validate_one_epoch(args, agent, policy, validation_nondom_list, best_f_list, validation_dataset_list, writer, test_batch, sample_solutions, epoch) 
        save_nes(policy, training_nondom_list, validation_nondom_list, best_f_list, epoch, args.title)
        if is_improving:
            save_nes(policy, training_nondom_list, validation_nondom_list, best_f_list, epoch, args.title, best=True)
            not_improving_count = 0
        else:
            not_improving_count += 1
        if not_improving_count == patience:
            break
        
if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)