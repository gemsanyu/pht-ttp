import copy
import os
import random
import subprocess
import sys

import numpy as np
from scipy.stats import wilcoxon
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from setup import setup
from ttp.ttp_dataset import TTPDataset, get_dataset_list
from ttp.ttp_env import TTPEnv
from utils import compute_single_loss, write_training_progress, write_validation_progress, write_test_progress, save
from utils import solve, prepare_args, write_test_progress, update_bp_only


def train_one_epoch(agent, critic, agent_opt, train_dataset_list, epoch, writer, entropy_loss_alpha=0.05):
    agent.train()
    critic.eval()
    sum_entropies_list = []
    agent_loss_list = []
    tour_lengths_list = []
    total_profits_list = []
    total_costs_list = []
    entropy_loss_list = []
    logprobs_list = []
    batch_size_per_dataset = int(args.batch_size/len(train_dataset_list))
    train_dataloader_list = [enumerate(DataLoader(train_dataset, batch_size=batch_size_per_dataset, shuffle=True, pin_memory=True)) for train_dataset in train_dataset_list]#, num_workers=4, pin_memory=True, shuffle=True)]
    # iterate until dataset empty, don't know elegant way to iterate yet
    is_done=False
    while not is_done:
        for dl_it in tqdm(train_dataloader_list, desc="Training"):
            try:
                batch_idx, batch = next(dl_it)    
                coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask,  best_profit_kp, best_route_length_tsp = batch
                env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask,  best_profit_kp, best_route_length_tsp)
                forward_results = solve(agent, env)
                tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = forward_results
                with torch.no_grad():
                    critic_forward_results = solve(critic, env)
                    _, _, critic_tour_lengths, critic_total_profits, critic_total_costs, _, _ = critic_forward_results
                agent_loss, entropy_loss, adv = compute_single_loss(total_costs, critic_total_costs, logprobs, sum_entropies)
                sum_entropies_list += [sum_entropies.detach().cpu().numpy()]
                agent_loss_list += [agent_loss.detach().cpu().numpy()[np.newaxis]]
                tour_lengths_list += [tour_lengths]
                total_profits_list += [total_profits]
                total_costs_list += [total_costs]
                entropy_loss_list += [entropy_loss.detach().cpu().numpy()[np.newaxis]]
                logprobs_list += [logprobs.detach().cpu().numpy()]
                loss = agent_loss + entropy_loss_alpha*entropy_loss
                loss.backward()
            except StopIteration:
                is_done=True
                break
        if not is_done:
            update_bp_only(agent, agent_opt)
            
    sum_entropies_list = np.concatenate(sum_entropies_list)
    agent_loss_list = np.concatenate(agent_loss_list)
    tour_lengths_list = np.concatenate(tour_lengths_list)
    total_profits_list = np.concatenate(total_profits_list)
    total_costs_list = np.concatenate(total_costs_list)
    entropy_loss_list = np.concatenate(entropy_loss_list)
    logprobs_list = np.concatenate(logprobs_list)
    write_training_progress(tour_lengths_list.mean(), total_profits_list.mean(), total_costs_list.mean(), agent_loss_list.mean(), entropy_loss_list.mean(), logprobs_list.mean(), sum_entropies_list.mean(), epoch, writer)

@torch.no_grad()
def validation_one_epoch(agent, critic, crit_total_cost_list, validation_dataset_list, test_env, epoch, writer):
    agent.eval()
    critic.eval()
    batch_size_per_dataset = int(args.batch_size/len(validation_dataset_list))
    validation_dataloader_list = [enumerate(DataLoader(validation_dataset, batch_size=batch_size_per_dataset, shuffle=False, pin_memory=True)) for validation_dataset in validation_dataset_list]#, num_workers=4, pin_memory=True, shuffle=False)]
    tour_length_list = []
    total_profit_list = []
    total_costs_list = []
    sum_entropies_list = []
    logprob_list = []
    is_done=False
    while not is_done:
        for dl_it in tqdm(validation_dataloader_list, desc="Validation"):
            try:
                batch_idx, batch = next(dl_it)
                coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
                env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask,  best_profit_kp, best_route_length_tsp)
                _, _, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, env)
                tour_length_list += [tour_lengths]
                total_profit_list += [total_profits]
                total_costs_list += [total_costs]
                sum_entropies_list += [sum_entropies.detach().cpu().numpy()]
                logprob_list += [logprobs]
            except StopIteration:
                is_done=True
                break
        
    # if there is no saved critic cost list then generate it/ first time
    if crit_total_cost_list is None:
        crit_total_cost_list = []
        validation_dataloader_list = [enumerate(DataLoader(validation_dataset, batch_size=batch_size_per_dataset, pin_memory=True, shuffle=False)) for validation_dataset in validation_dataset_list]#, num_workers=4, pin_memory=True, shuffle=False)]
        is_done=False
        while not is_done:
            for dl_it in tqdm(validation_dataloader_list, desc="Crit Validation Generate"):
                try:    
                    batch_idx, batch = next(dl_it)
                    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask,  best_profit_kp, best_route_length_tsp = batch
                    env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask,  best_profit_kp, best_route_length_tsp)
                    _, _, crit_tour_lengths, crit_total_profits, crit_total_costs, _, _ = solve(critic, env)
                    crit_total_cost_list += [crit_total_costs]
                except StopIteration:
                    is_done=True
                    break
            # crit_tour_length_list += [crit_tour_lengths]
            # crit_total_profit_list += [crit_total_profits]
        crit_total_cost_list = np.concatenate(crit_total_cost_list)
    total_costs_list = np.concatenate(total_costs_list)
    tour_length_list = np.concatenate(tour_length_list)
    total_profit_list = np.concatenate(total_profit_list)
    sum_entropies_list = np.concatenate(sum_entropies_list)
    # crit_tour_length_list = np.concatenate(crit_tour_length_list) 
    # crit_total_profit_list = np.concatenate(crit_total_profit_list) 
    mean_tour_length = tour_length_list.mean()
    mean_total_profit = total_profit_list.mean()
    mean_total_cost = total_costs_list.mean()
    mean_entropies = sum_entropies_list.mean()
    mean_logprob = torch.cat(logprob_list).mean()
    # print(len(total_costs_list), len(crit_total_cost_list))
    write_validation_progress(mean_tour_length, mean_total_profit, mean_total_cost, mean_entropies, mean_logprob, epoch, writer)
    
    #check if agent better than critic now?
    res = wilcoxon(total_costs_list, crit_total_cost_list, alternative="greater")
    print("-----------------Validation pvalue:", res.pvalue)
    is_improving=False
    if res.pvalue < 0.05:
        is_improving = True
        critic.load_state_dict(copy.deepcopy(agent.state_dict()))
        crit_total_cost_list = total_costs_list
    #test?
    _, _, tour_lengths, total_profits, total_costs, _, _ = solve(agent, test_env)
    write_test_progress(tour_lengths.mean(), total_profits.mean(), total_costs.mean(), logprobs.mean(), writer)
    return is_improving, crit_total_cost_list

def run(args):
    patience=20
    not_improving_count = 0
    agent, agent_opt, critic, crit_total_cost_list, last_epoch, writer, test_env = setup(args)
    nn_list = [10,20,30,40,50]
    nipc_list = [1,3,5,10]
    len_types = len(nn_list)*len(nipc_list)
    train_num_samples_per_dataset = int(args.num_training_samples/len_types)
    validation_num_samples_per_dataset = int(args.num_validation_samples/len_types)
    train_dataset_list = get_dataset_list(train_num_samples_per_dataset, nn_list, nipc_list, mode="training")
    validation_dataset_list = get_dataset_list(validation_num_samples_per_dataset, nn_list, nipc_list, mode="validation")

    for epoch in tqdm(range(last_epoch, args.max_epoch), desc="EPOCH:"):
        train_one_epoch(agent, critic, agent_opt, train_dataset_list, epoch, writer)
        is_improving, crit_total_cost_list  = validation_one_epoch(agent, critic, crit_total_cost_list, validation_dataset_list, test_env, epoch, writer)
        save(agent, agent_opt, critic, crit_total_cost_list, args.title, epoch )
        if is_improving:
            save(agent, agent_opt, critic, crit_total_cost_list, args.title, epoch, is_best=True)
            not_improving_count = 0
        else:
            not_improving_count += 1
        if not_improving_count == patience:
            break 

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)
