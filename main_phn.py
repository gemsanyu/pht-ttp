import os
import random
import subprocess
import sys

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import get_parser
from setup_phn import setup_phn
from solver.hv_maximization import HvMaximization
from ttp.ttp_dataset import TTPDataset, combine_batch_list
from ttp.ttp_env import TTPEnv
from utils import update_phn, save_phn, solve_decode_only, encode, write_test_phn_progress
from validator import load_validator
from validate_phn import test_one_epoch
from policy.non_dominated_sorting import fast_non_dominated_sort

CPU_DEVICE = torch.device("cpu")
MAX_PATIENCE = 50

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def decode_one_batch(agent, param_dict_list, train_env, static_embeddings):
    pop_size = len(param_dict_list)
    batch_size = train_env.batch_size
    travel_time_list = torch.zeros((pop_size, batch_size), dtype=torch.float32)
    total_profit_list = torch.zeros((pop_size, batch_size), dtype=torch.float32)
    logprobs_list = torch.zeros((pop_size, batch_size), dtype=torch.float32, device=agent.device)
    
    for n, param_dict in enumerate(param_dict_list):
        solve_output = solve_decode_only(agent, train_env, static_embeddings, param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        travel_time_list[n,:] = tour_lengths
        total_profit_list[n,:] = total_profits
        logprobs_list[n,:] = logprobs
    inv_total_profit_list = -total_profit_list 
    f_list = torch.cat((travel_time_list.unsqueeze(2),inv_total_profit_list.unsqueeze(2)), dim=-1)
    return f_list, logprobs_list


def solve_one_batch(agent, param_dict_list, batch):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp = batch
    train_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp)
    static_features, dynamic_features, eligibility_mask = train_env.begin()
    num_nodes, num_items, batch_size = train_env.num_nodes, train_env.num_items, train_env.batch_size
    static_embeddings = encode(agent, static_features, num_nodes, num_items, batch_size)

    # sample rollout
    agent.train()
    f_list, logprobs_list = decode_one_batch(agent, param_dict_list, train_env, static_embeddings)
    return f_list, logprobs_list

def compute_spread_loss(logprobs, f_list, param_dict_list):
    # nadir_points,_ = torch.max(f_list, dim=0, keepdim=True)
    # utopia_points,_ = torch.min(f_list, dim=0, keepdim=True)
    # diff = (nadir_points-utopia_points)+1e-8
    # norm_f_list = (f_list-utopia_points)/diff
    param_list = [param_dict["v1"].ravel().unsqueeze(0) for param_dict in param_dict_list]
    param_list = torch.cat(param_list).unsqueeze(0)
    # norm_f_list = f_list
    # batched_nf = torch.transpose(norm_f_list,0,1)
    # distance_matrix = torch.cdist(batched_nf, batched_nf)
    distance_matrix = torch.cdist(param_list, param_list)
    batched_distance_per_ray = torch.transpose(distance_matrix.sum(dim=2),0,1)
    # spread_loss = (batched_distance_per_ray*logprobs).mean()
    spread_loss = batched_distance_per_ray.mean()
    return spread_loss

def compute_mimic_loss(phn, param_dict_list, f_list):
    num_ray, batch_size, _ = f_list.shape
    param_list = [param_dict["v1"].detach().clone().ravel() for param_dict in param_dict_list]
    nadir_points,_ = torch.max(f_list, dim=0, keepdim=True)
    utopia_points,_ = torch.min(f_list, dim=0, keepdim=True)
    diff = (nadir_points-utopia_points)+1e-8
    norm_f_list = (f_list-utopia_points)/diff
    # print(norm_f_list, norm_f_list.sum(dim=-1, keepdim=True))
    
    norm_f_list /= (norm_f_list.sum(dim=-1, keepdim=True) +1e-8)
    
    loss = []
    for b_idx in range(batch_size):
        nondom_idx = fast_non_dominated_sort(norm_f_list[:, b_idx].numpy())[0]
        for r_idx in nondom_idx:
            ray = norm_f_list[r_idx, b_idx]
            out = phn(ray)
            param = param_list[r_idx]
            pred_param  = out["v1"].ravel()
            loss += [torch.norm(pred_param-param).unsqueeze(0)]
    loss = torch.cat(loss).mean()
            # print(ray,"bsbsb")
            # print(param,pred_param,"ABC")
    print(loss)
    return loss

    
def train_one_batch(agent, phn, phn_opt, batch, writer, num_ray=16, ld=1):
    agent.train()
    mo_opt = HvMaximization(n_mo_sol=num_ray, n_mo_obj=2)
    ray_list = []

    param_dict_list = []
    for i in range(num_ray):
        start, end = 0.1, np.pi/2-0.1
        r = np.random.uniform(start + i*(end-start)/num_ray, start+ (i+1)*(end-start)/num_ray)
        ray = np.array([np.cos(r),np.sin(r)], dtype='float32')
        ray /= ray.sum()
        ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2))
        ray = torch.from_numpy(ray).to(agent.device)
        param_dict = phn(ray)
        param_dict_list += [param_dict]
        ray_list += [ray]
    ray_list = torch.stack(ray_list)
    
    f_list, logprob_list = solve_one_batch(agent, param_dict_list, batch)
    with torch.no_grad():
        agent.eval()
        greedy_f_list, _ = solve_one_batch(agent, param_dict_list, batch)
    
    
    adv_list = (f_list-greedy_f_list).to(agent.device)
    adv_max, _ = torch.max(adv_list, dim=0, keepdim=True)
    adv_min, _ = torch.min(adv_list, dim=0, keepdim=True)
    adv_max, adv_min = adv_max.detach(), adv_min.detach()
    adv_list = (adv_list-adv_min)/(adv_max-adv_min+1e-8)
    logprob_list_exp = logprob_list.unsqueeze(2).expand_as(f_list)
    loss = (adv_list)*logprob_list_exp
    loss_max, _ = torch.max(loss, dim=0, keepdim=True)
    loss_min, _ = torch.min(loss, dim=0, keepdim=True)
    loss_max, loss_min = loss_max.detach(), loss_min.detach()
    norm_loss = (loss-loss_min)/(loss_max-loss_min+1e-8)
    norm_obj = norm_loss.detach().cpu().numpy()
    _, num_instances, _ = norm_obj.shape
    hv_drv_list = []
    for i in range(num_instances):
        obj_instance = np.transpose(norm_obj[:, i, :]) 
        hv_drv_instance = mo_opt.compute_weights(obj_instance).transpose(0,1).unsqueeze(1)
        hv_drv_list.append(hv_drv_instance)
    hv_drv_list = torch.cat(hv_drv_list, dim=1).to(agent.device)
    losses_per_obj = loss*hv_drv_list
    losses_per_instance = torch.sum(losses_per_obj, dim=2)
    losses_per_ray = torch.mean(losses_per_instance, dim=1)
    total_loss = torch.sum(losses_per_ray)
    # total_loss = 0
    # # compute cosine similarity penalty
    # total_loss += 0.01*mimic_loss - 0.01*spread_loss
    # mimic_loss = compute_mimic_loss(phn, param_dict_list, f_list)
    # spread_loss = compute_spread_loss(logprob_list, f_list, param_dict_list)
    
    # total_loss -= 0.1*spread_loss
    # print(spread_loss)
    # print(mimic_loss, spread_loss)

    # print("A")
    cos_penalty = cosine_similarity(loss, ray_list.unsqueeze(1), dim=2)
    cos_penalty_per_ray = cos_penalty.mean(dim=1)
    total_loss -= ld*cos_penalty_per_ray.sum()
    # replace cosine similarity with 


    update_phn(phn, phn_opt, total_loss)
    agent.zero_grad(set_to_none=True)
    phn.zero_grad(set_to_none=True)
    # write_training_phn_progress(writer, loss.detach().cpu(),ray_list.cpu(),cos_penalty.detach().cpu())


def train_one_epoch(agent, phn, phn_opt, writer, batch_size, total_num_samples, num_ray, ld):
    # phn.train()
    dataset = TTPDataset(total_num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for idx, batch in tqdm(enumerate(dataloader)):
        train_one_batch(agent, phn, phn_opt, batch, writer, num_ray, ld)
        
def initialize(target_param,phn,opt,tb_writer):
    ray = np.asanyarray([[0.5,0.5]],dtype=float)
    # i = random.random()*100
    # start, end = 0.1, np.pi/2-0.1
    # r = np.random.uniform(start + i*(end-start)/100, start+ (i+1)*(end-start)/100)
    # ray = np.array([np.cos(r),np.sin(r)], dtype='float32')
    # ray /= ray.sum()
    ray = torch.from_numpy(ray).to(phn.device, dtype=torch.float32)
    param_dict = phn(ray)
    v = (param_dict["v1"]).ravel()
    # qe1 = (param_dict["qe1_weight"]).ravel()
    # param = torch.cat([v,qe1])
    param=v
    loss = torch.norm(target_param-param)
    opt.zero_grad(set_to_none=True)
    tb_writer.add_scalar("Initialization loss", loss.cpu().item())
    loss.backward()
    opt.step()
    return loss.cpu().item()

def init_phn_output(agent, phn, tb_writer, max_step=1000):
    v,fe1,qe1 = None,None,None
    for name, param in agent.named_parameters():
        if name == "pointer.attention_layer.v":
            v = param.data.ravel()
        # elif name == "pointer.attention_layer.features_embedder.weight":
        #     fe1 = param.data.ravel()
        elif name == "pointer.attention_layer.query_embedder.weight":
            qe1 = param.data.ravel()
        
    v = v.detach().clone()
    # fe1 = fe1.detach().clone()
    # qe1 = qe1.detach().clone()
    # target_param = torch.cat([v,qe1])
    target_param=v
    opt_init = torch.optim.Adam(phn.parameters(), lr=1e-4)
    for i in range(max_step):
        loss = initialize(target_param,phn,opt_init,tb_writer)
        if loss < 1e-3:
            break
    return phn

def run(args):
    agent, phn, phn_opt, last_epoch, writer, checkpoint_path, test_env, test_sample_solutions = setup_phn(args)
    # vd_proc_cmd = ["python",
    #                 "validate_phn.py",
    #                 "--ray-hidden-size",
    #                 str(args.ray_hidden_size),
    #                 "--title",
    #                 args.title,
    #                 "--dataset-name",
    #                 args.dataset_name,
    #                 "--device",
    #                 "cpu"]
    # vd_proc = subprocess.Popen(vd_proc_cmd)
    # test_solution_list = test_one_epoch(agent, phn, test_env)
    # write_test_phn_progress(writer, test_solution_list, 0, test_sample_solutions)
    early_stop = 0
    # initialization
    if last_epoch == 0:
        phn = init_phn_output(agent, phn, writer, max_step=1000)
        # vd_proc.wait()
        save_phn(phn, phn_opt, 0, args.title)
        # vd_proc = subprocess.Popen(vd_proc_cmd)
    for epoch in range(last_epoch, args.max_epoch):
        train_one_epoch(agent, phn, phn_opt, writer, args.batch_size, args.num_training_samples, args.num_ray, args.ld)
        if epoch % 5 == 0:
            test_solution_list = test_one_epoch(agent, phn, test_env)
            write_test_phn_progress(writer, test_solution_list, epoch, test_sample_solutions)
    
        # vd_proc.wait()
        # vd = load_validator(args.title)
        # if vd.is_improving:
        #     early_stop = 0
        #     save_phn(phn, phn_opt, epoch, args.title, best=True)
        # else:   
        #     early_stop += 1
        save_phn(phn, phn_opt, epoch, args.title)
        # vd_proc = subprocess.Popen(vd_proc_cmd)
    # vd_proc.wait()

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)