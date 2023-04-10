import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from setup_phn import setup_phn
from ttp.ttp_dataset import TTPDataset
from utils import update_phn, save_phn
from utils import prepare_args
from utils_moo import init_phn_output, validate_one_epoch, write_training_phn_progress, compute_loss
from utils_moo import compute_spread_loss, generate_params, solve_one_batch, generate_rays
from utils_moo import solve_one_batchv2, validate_one_epochv2
from validator import load_validator
    
def train_one_batch(agent, phn, phn_opt, batch, writer, num_ray=16, ld=1, is_initialize=False):
    agent.train()
    # ray_list = generate_rays(num_ray, phn.device)
    ray_list, param_dict_list = generate_params(phn, num_ray, phn.device)
    logprob_list, f_list, sum_entropies_list = solve_one_batch(agent, param_dict_list, batch)
    # logprob_list, f_list, sum_entropies_list, param_dict_list = solve_one_batchv2(agent, phn, ray_list, batch)
    with torch.no_grad():
        agent.eval()
        _, greedy_f_list, _ = solve_one_batch(agent, param_dict_list, batch)
        # _, greedy_f_list, _, _ = solve_one_batchv2(agent, phn, ray_list, batch)
    loss_obj, cos_penalty_loss = compute_loss(logprob_list, f_list, greedy_f_list,ray_list)
    total_loss = loss_obj
    if is_initialize:
        # total_loss = -0.1*spread_loss
        total_loss = 0
    spread_loss = compute_spread_loss(logprob_list, f_list, param_dict_list)
    total_loss -= 0.1*spread_loss
    total_loss += ld*cos_penalty_loss
    
    update_phn(phn, phn_opt, total_loss)
    agent.zero_grad(set_to_none=True)
    phn.zero_grad(set_to_none=True)
    write_training_phn_progress(writer, loss_obj.detach().cpu(), cos_penalty_loss.detach().cpu())

def train_one_epoch(args, agent, phn, phn_opt, writer, training_dataset, is_initialize):
    phn.train()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    for idx, batch in tqdm(enumerate(dataloader)):
        train_one_batch(agent, phn, phn_opt, batch, writer, args.num_ray, args.ld, is_initialize)
        
def run(args):
    agent, phn, phn_opt, last_epoch, writer, test_batch, test_sample_solutions = setup_phn(args)
    validator = load_validator(args)
    training_dataset = TTPDataset(num_samples=args.num_training_samples)
    validation_dataset = TTPDataset(num_samples=args.num_validation_samples)
    # if last_epoch == 0:
    #     init_phn_output(agent, phn, writer, max_step=1000)
    #     validate_one_epochv2(args, agent, phn, validator, validation_dataset,test_batch,test_sample_solutions, writer, -1)  
    #     save_phn(phn, phn_opt, -1, args.title)
    for epoch in range(last_epoch, args.max_epoch):
        if epoch <=10:
            train_one_epoch(args, agent, phn, phn_opt, writer, training_dataset, is_initialize=True)
        else:
            train_one_epoch(args, agent, phn, phn_opt, writer, training_dataset, is_initialize=False)
        if epoch % 5 == 0:
            validate_one_epoch(args, agent, phn, validator, validation_dataset,test_batch,test_sample_solutions, writer, epoch) 
        save_phn(phn, phn_opt, epoch, args.title)

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)