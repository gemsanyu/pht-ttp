import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from arguments import get_parser
from setup import setup
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv
from utils import compute_loss, update, write_training_progress, write_validation_progress, write_test_progress, save
from utils import solve

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def train_one_epoch(agent, agent_opt, train_dataset, writer, critic_alpha=0.8, entropy_loss_alpha=0.05):
    agent.train()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    critic_costs = None
    for batch_idx, batch in (enumerate(train_dataloader), desc="step", position=1):
    # for _, batch in enumerate(train_dataloader):
        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
        env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, env)
        if critic_costs is None:
            critic_costs = total_costs.mean()
        else:
            critic_costs = critic_alpha*critic_costs + (1-critic_alpha)*total_costs.mean()
        agent_loss, entropy_loss = compute_loss(total_costs, critic_costs, logprobs, sum_entropies)
        loss = agent_loss + entropy_loss_alpha*entropy_loss
        update(agent, agent_opt, loss)
        write_training_progress(tour_lengths.mean(), total_profits.mean(), total_costs.mean(), agent_loss.detach(), entropy_loss.detach(), critic_costs, logprobs.detach().mean(), env.num_nodes, env.num_items, writer)

@torch.no_grad()
def validation_one_epoch(agent, validation_dataset, writer):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=0)
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
    agent, agent_opt, last_epoch, writer, checkpoint_path, test_env = setup(args)
    validation_size = int(0.1*args.num_training_samples)
    training_size = args.num_training_samples - validation_size
    num_nodes_list = [50]
    num_items_per_city_list = [3]
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
        train_one_epoch(agent, agent_opt, train_dataset, writer)
        validation_cost = validation_one_epoch(agent, validation_dataset, writer)
        test_one_epoch(agent, test_env, writer)
        save(agent, agent_opt, validation_cost, epoch, checkpoint_path)

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)