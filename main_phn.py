from cgi import test
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


from arguments import get_parser
from setup import setup_phn
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv
from utils import update_phn, write_test_phn_progress, write_training_phn_progress, save_phn
from utils import solve

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def train_one_epoch(agent, phn, phn_opt, train_dataset, writer, critic_alpha=0.8, alpha=0.2):
    agent.train()
    phn.train()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    critic_profits, critic_tour_lengths = None, None
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training", position=1):
    # for batch_idx, batch in enumerate(train_dataloader): 
        # sample a ray preference
        ray = torch.from_numpy(
                np.random.dirichlet([alpha, alpha], 1).astype(np.float32).flatten()
            ).to(agent.device)
        ray = ray.unsqueeze(0)
        param_dict = phn(ray)

        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
        env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, env, param_dict)
        norm_tour_lengths = tour_lengths/env.best_route_length_tsp - 1.
        norm_total_profits = 1. - total_profits/env.best_profit_kp
        norm_tour_lengths = norm_tour_lengths.to(agent.device)
        norm_total_profits = norm_total_profits.to(agent.device)
        tour_length_loss = (logprobs*norm_tour_lengths).mean()
        profit_loss = (logprobs*norm_total_profits).mean()

        loss = torch.stack([tour_length_loss, profit_loss])
        epo_loss = (ray.squeeze(0)*loss).sum()
        update_phn(phn, phn_opt, epo_loss)
        agent.zero_grad(set_to_none=True)
        write_training_phn_progress(total_profits.mean(), tour_lengths.mean(), profit_loss.detach(), tour_length_loss.detach(), epo_loss.detach().cpu(), logprobs.detach().cpu().mean(), env.num_nodes, env.num_items, writer)

@torch.no_grad()
def test_one_epoch(agent, phn, test_env, test_sample_solutions, writer, epoch, n_solutions=100):
    agent.eval()
    phn.eval()
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    solution_list = []
    for ray in tqdm(ray_list, desc="Testing"):
        param_dict = phn(ray.to(agent.device))
        tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve(agent, test_env, param_dict)
        solution_list += [torch.stack([tour_length, total_profit], dim=1)]
    solution_list = torch.cat(solution_list)
    write_test_phn_progress(writer, solution_list, epoch, test_sample_solutions)

def run(args):
    agent, phn, phn_opt, last_epoch, writer, checkpoint_path, test_env, test_sample_solutions = setup_phn(args)
    num_nodes_list = [50]
    num_items_per_city_list = [1,3,5]
    config_list = [(num_nodes, num_items_per_city) for num_nodes in num_nodes_list for num_items_per_city in num_items_per_city_list]
    num_configs = len(num_nodes_list)*len(num_items_per_city_list)
    for epoch in range(last_epoch, args.max_epoch):
        print("EPOCH:", epoch)
        print("---------------------------------------")
        config_it = epoch%num_configs
        if config_it == 0:
            random.shuffle(config_list)
        num_nodes, num_items_per_city = config_list[config_it]
        dataset = TTPDataset(args.num_training_samples, num_nodes, num_items_per_city)
        train_one_epoch(agent, phn, phn_opt, dataset, writer)
        test_one_epoch(agent, phn, test_env, test_sample_solutions, writer, epoch)
        save_phn(phn, phn_opt, epoch, checkpoint_path)

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    args = prepare_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)