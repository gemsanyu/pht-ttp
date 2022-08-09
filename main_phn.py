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
from solver import EPOSolver
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv
from utils import solve, compute_multi_loss, update_phn, write_test_phn_progress, write_training_phn_progress, save_phn

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def train_one_epoch(agent, phn, phn_opt, solver, train_dataset, writer, critic_alpha=0.8, alpha=0.2):
    agent.eval()
    phn.train()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    critic_profits, critic_tour_lengths = None, None
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training", position=1):
        # sample a ray preference
        ray = torch.from_numpy(
                np.random.dirichlet([alpha, alpha], 1).astype(np.float32).flatten()
            ).to(agent.device)
        ray = ray.unsqueeze(0)
        # generate parameters
        param_dict = phn(ray)

        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask = batch
        env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, env, param_dict, normalized=True)
        remaining_profits = 1.-total_profits
        profit_loss, tour_length_loss = compute_multi_loss(remaining_profits, tour_lengths, logprobs)
        loss = torch.stack([tour_length_loss, profit_loss])
        epo_loss = solver(loss, ray.squeeze(0), list(phn.parameters()))
        update_phn(phn, phn_opt, epo_loss)
        agent.zero_grad(set_to_none=True)
        write_training_phn_progress(total_profits.mean(), tour_lengths.mean(), profit_loss.detach(), tour_length_loss.detach(), epo_loss.detach(), logprobs.detach().mean(), env.num_nodes, env.num_items, writer)

@torch.no_grad()
def validation_one_epoch(agent, validation_dataset, writer):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=4)
    tour_length_list = []
    total_profit_list = []
    total_cost_list = []
    logprob_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc="step", position=1):
        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask = batch
        env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve(agent, env)
        tour_length_list += [tour_lengths]
        total_profit_list += [total_profits]
        total_cost_list += [total_costs]
        logprob_list += [logprobs]
    mean_tour_length = torch.cat(tour_length_list).mean()
    mean_total_profit = torch.cat(total_profit_list).mean()
    mean_total_cost = torch.cat(total_cost_list).mean()
    mean_logprob = torch.cat(logprob_list).mean()
    return mean_total_cost


@torch.no_grad()
def test_one_epoch(agent, phn, test_env, writer, epoch, n_solutions=100):
    agent.eval()
    phn.eval()
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    solution_list = []
    for ray in tqdm(ray_list, desc="Testing"):
        param_dict = phn(ray.to(agent.device))
        tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve(agent, test_env, param_dict, normalized=False)
        solution_list += [torch.stack([tour_length, total_profit], dim=1)]
    solution_list = torch.cat(solution_list)
    write_test_phn_progress(writer, solution_list, epoch)

def run(args):
    agent, phn, phn_opt, solver, last_epoch, writer, checkpoint_path, test_env = setup_phn(args)
    validation_size = int(0.1*args.num_training_samples)
    training_size = args.num_training_samples - validation_size
    num_nodes_list = [50, 100]
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
        train_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])
        train_one_epoch(agent, phn, phn_opt, solver, train_dataset, writer)
        # validation_cost = validation_one_epoch(agent, validation_dataset, writer)
        test_one_epoch(agent, phn, test_env, writer, epoch)
        save_phn(phn, phn_opt, epoch, checkpoint_path)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    args = prepare_args()
    torch.set_num_threads(os.cpu_count())
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)