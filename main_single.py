import copy
import os
import random
import subprocess
import sys

import numpy as np
from scipy.stats import wilcoxon
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from setup import setup
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv
from utils import compute_single_loss, update, write_training_progress, write_validation_progress, write_test_progress, save
from utils import solve, prepare_args, write_test_progress


def train_one_epoch(agent, critic, agent_opt, train_dataset, epoch, writer, entropy_loss_alpha=0.05):
    agent.train()
    critic.eval()
    sum_entropies_list = []
    agent_loss_list = []
    tour_lengths_list = []
    total_profits_list = []
    total_costs_list = []
    entropy_loss_list = []
    logprobs_list = []

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="step", position=1):
        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp = batch
        env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp)
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
        # print(agent_loss_list)
        update(agent, agent_opt, loss)
    sum_entropies_list = np.concatenate(sum_entropies_list)
    agent_loss_list = np.concatenate(agent_loss_list)
    tour_lengths_list = np.concatenate(tour_lengths_list)
    total_profits_list = np.concatenate(total_profits_list)
    total_costs_list = np.concatenate(total_costs_list)
    entropy_loss_list = np.concatenate(entropy_loss_list)
    logprobs_list = np.concatenate(logprobs_list)
    write_training_progress(tour_lengths_list.mean(), total_profits_list.mean(), total_costs_list.mean(), agent_loss_list.mean(), entropy_loss_list.mean(), logprobs_list.mean(), sum_entropies_list.mean(), epoch, writer)

@torch.no_grad()
def validation_one_epoch(agent, critic, crit_total_cost_list, validation_dataset, test_env, epoch, writer):
    agent.eval()
    critic.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    tour_length_list = []
    total_profit_list = []
    total_costs_list = []
    sum_entropies_list = []
    logprob_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc="step", position=1):
        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp = batch
        env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp)
        _, _, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, env)
        tour_length_list += [tour_lengths]
        total_profit_list += [total_profits]
        total_costs_list += [total_costs]
        sum_entropies_list += [sum_entropies]
        logprob_list += [logprobs]
    # if there is no saved critic cost list then generate it/ first time
    if crit_total_cost_list is None:
        crit_total_costs_list = []
        for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc="step", position=1):
            coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp = batch
            env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp)
            _, _, crit_tour_lengths, crit_total_profits, crit_total_costs, _, _ = solve(critic, env)
            crit_total_costs_list += [crit_total_costs]
            # crit_tour_length_list += [crit_tour_lengths]
            # crit_total_profit_list += [crit_total_profits]
        crit_total_costs_list = np.concatenate(crit_total_costs_list)
           
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
    write_validation_progress(mean_tour_length, mean_total_profit, mean_total_cost, mean_entropies, mean_logprob, epoch, writer)
    
    #check if agent better than critic now?
    res = wilcoxon(total_costs_list, crit_total_costs_list, alternative="greater")
    print("Validation pvalue:", res.pvalue)
    is_improving=False
    if res.pvalue < 0.05:
        is_improving = True
        critic.load_state_dict(copy.deepcopy(agent.state_dict()))
    
    #test?
    _, _, tour_lengths, total_profits, total_costs, _, _ = solve(agent, test_env)
    write_test_progress(tour_lengths.mean(), total_profits.mean(), total_costs.mean(), logprobs.mean(), writer)
    return is_improving, crit_total_cost_list

def run(args):
    patience=10
    not_improving_count = 0
    agent, agent_opt, critic, crit_total_cost_list, last_epoch, writer, test_env = setup(args)
    training_dataset = TTPDataset(args.num_training_samples, mode="training")
    validation_dataset = TTPDataset(args.num_validation_samples, mode="validation")
    for epoch in range(last_epoch, args.max_epoch):
        train_one_epoch(agent, critic, agent_opt, training_dataset, epoch, writer)
        is_improving, crit_total_cost_list  = validation_one_epoch(agent, critic, crit_total_cost_list, validation_dataset, test_env, epoch, writer)
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
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)