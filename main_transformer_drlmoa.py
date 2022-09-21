import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from arguments import get_parser
from setup import setup_drlmoa
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv
from utils import compute_loss, update, write_training_progress, write_validation_progress, write_test_progress, save
from transformer_utils import solve

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def train_one_epoch(agent, agent_opt, train_dataset, writer, ray):
    agent.train()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="step", position=1):
        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
        env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, env)
        norm_tour_lengths = tour_lengths/env.best_route_length_tsp - 1.
        norm_total_profits = 1. - total_profits/env.best_profit_kp
        norm_tour_lengths = norm_tour_lengths.to(agent.device)
        norm_total_profits = norm_total_profits.to(agent.device)
        tour_length_loss = (logprobs*norm_tour_lengths).mean()
        profit_loss = (logprobs*norm_total_profits).mean()
        loss = torch.stack([tour_length_loss, profit_loss])
        agent_loss = (ray*loss).sum()
        update(agent, agent_opt, agent_loss)
        write_training_progress(tour_lengths.mean(), total_profits.mean(), total_costs.mean(), agent_loss.detach(), 0, 0, logprobs.detach().mean(), env.num_nodes, env.num_items, writer)

@torch.no_grad()
def validation_one_epoch(agent, validation_dataset, writer):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    tour_length_list = []
    total_profit_list = []
    total_cost_list = []
    logprob_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc="step", position=1):
        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
        env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, env)
        tour_length_list += [tour_lengths]
        total_profit_list += [total_profits]
        total_cost_list += [total_costs]
        logprob_list += [logprobs]
    mean_tour_length = torch.cat(tour_length_list).mean()
    mean_total_profit = torch.cat(total_profit_list).mean()
    mean_total_cost = torch.cat(total_cost_list).mean()
    mean_logprob = torch.cat(logprob_list).mean()
    write_validation_progress(mean_tour_length, mean_total_profit, mean_total_cost, mean_logprob, writer)
    return mean_total_cost


@torch.no_grad()
def test_one_epoch(agent, test_env, writer):
    agent.eval()
    tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve(agent, test_env)
    write_test_progress(tour_length, total_profit, total_cost, logprob, writer)    
        

def run(args):
    agent, agent_opt, last_epoch, writer, checkpoint_path, test_env = setup_drlmoa(args)
    a = (args.weight_idx-1.)/(args.total_weight-1.)
    b = 1-a
    ray = torch.tensor([a,b], device=args.device)
    validation_size = int(0.1*args.num_training_samples)
    training_size = args.num_training_samples - validation_size
    num_nodes_list = [50]
    num_items_per_city_list = [1,3,5]
    config_list = [(num_nodes, num_items_per_city) for num_nodes in num_nodes_list for num_items_per_city in num_items_per_city_list]
    num_configs = len(num_nodes_list)*len(num_items_per_city_list)
    for epoch in range(last_epoch, args.max_epoch):
        config_it = epoch%num_configs
        if config_it == 0:
            random.shuffle(config_list)
        num_nodes, num_items_per_city = config_list[config_it]
        print("EPOCH:", epoch, "NN:", num_nodes, "NIC:", num_items_per_city)
        dataset = TTPDataset(args.num_training_samples, num_nodes, num_items_per_city)
        train_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])
        train_one_epoch(agent, agent_opt, train_dataset, writer, ray)
        # validation_cost = validation_one_epoch(agent, validation_dataset, writer)
        # test_one_epoch(agent, test_env, writer)
        save(agent, agent_opt, 0, epoch, checkpoint_path)

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(os.cpu_count())
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)