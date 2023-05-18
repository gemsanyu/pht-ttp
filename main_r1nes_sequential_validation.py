import math
import subprocess
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from agent.agent import Agent
from setup_r1nes import setup_r1_nes
from ttp.ttp_dataset import TTPDataset
from policy.r1_nes import R1_NES
from policy.utils import get_score_hv_contributions
from utils import save_nes, prepare_args, CPU_DEVICE
from utils_moo import solve_one_batch, validate_one_epoch
from validator import load_validator, save_validator


def train_one_batch(agent:Agent, policy: R1_NES, batch, pop_size):
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    f_list = solve_one_batch(agent, param_dict_list, batch) 
    _, batch_size, _ = f_list.shape
    score = []
    for batch_idx in range(batch_size):
        batch_score = get_score_hv_contributions(f_list[:,batch_idx,:], policy.negative_hv)    
        score += [batch_score]
    score = torch.cat(score, dim=-1)
    score = score.mean(dim=1, keepdim=True)
    x_list = sample_list - policy.mu
    w_list = x_list/math.exp(policy.ld)
    policy.update(w_list, x_list, score)


@torch.no_grad()
def train_one_generation(args, agent:Agent, policy: R1_NES, training_dataset, pop_size=10):
    agent.eval()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    for i, batch in tqdm(enumerate(dataloader), desc="Generation"):
        train_one_batch(agent, policy, batch, pop_size)
    
def run(args):
    agent, policy, last_epoch, writer, test_batch, test_sample_solutions = setup_r1_nes(args)
    validator = load_validator(args)
    training_dataset = TTPDataset(num_samples=args.num_training_samples, mode="training")
    validation_dataset = TTPDataset(num_samples=args.num_validation_samples, mode="validation")
    for epoch in range(last_epoch, args.max_epoch):
        train_one_generation(args, agent, policy, training_dataset, pop_size=policy.pop_size)
        policy.write_progress_to_tb(writer, epoch)
        validate_one_epoch(args, agent, policy, validator, validation_dataset, test_batch, test_sample_solutions, writer, epoch)
        save_nes(policy, epoch, args.title)
        if validator.is_improving:
            save_nes(policy, epoch,args.title, True)
        save_validator(validator, args.title)

if __name__ == '__main__':
    args = prepare_args()
    # torch.set_num_threads(os.cpu_count())
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)