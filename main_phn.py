import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from arguments import get_parser
from setup import setup_phn
from solver.hv_maximization import HvMaximization
from ttp.ttp_dataset import TTPDataset, combine_batch_list
from ttp.ttp_env import TTPEnv
from utils import update_phn, write_training_phn_progress, save_phn
from utils import solve_decode_only, encode
from validator import load_validator


CPU_DEVICE = torch.device("cpu")
MAX_PATIENCE = 50

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def decode_one_batch(agent, param_dict_list, train_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static):
    pop_size = len(param_dict_list)
    batch_size = train_env.batch_size
    travel_time_list = torch.zeros((pop_size, batch_size), dtype=torch.float32)
    total_profit_list = torch.zeros((pop_size, batch_size), dtype=torch.float32)
    logprobs_list = torch.zeros((pop_size, batch_size), dtype=torch.float32, device=agent.device)
    
    for n, param_dict in enumerate(param_dict_list):
        solve_output = solve_decode_only(agent, train_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static, param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        travel_time_list[n,:] = tour_lengths
        total_profit_list[n,:] = total_profits
        logprobs_list[n,:] = logprobs
    inv_total_profit_list = -total_profit_list 
    f_list = torch.cat((travel_time_list.unsqueeze(2),inv_total_profit_list.unsqueeze(2)), dim=-1)
    return f_list, logprobs_list


def solve_one_batch(agent, param_dict_list, batch):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
    train_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    static_features = train_env.get_static_features()    
    num_nodes, num_items, batch_size = train_env.num_nodes, train_env.num_items, train_env.batch_size
    encode_output = encode(agent, static_features, num_nodes, num_items, batch_size)
    static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static = encode_output
    # sample rollout
    agent.train()
    f_list, logprobs_list = decode_one_batch(agent, param_dict_list, train_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
    # greedy critic rollout
    agent.eval()
    with torch.no_grad():
        critic_f_list, _ = decode_one_batch(agent, param_dict_list, train_env, static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static)
    return f_list, critic_f_list, logprobs_list

def train_one_batch(agent, phn, phn_opt, batch_list, writer, num_ray=16, ld=1):
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
    
    all_f_list = []
    all_crit_f_list = []
    all_logprob_list = []
    for batch in batch_list:
        f_list, crit_f_list, logprob_list = solve_one_batch(agent, param_dict_list, batch)
        all_f_list += [f_list]
        all_crit_f_list += [crit_f_list]
        all_logprob_list += [logprob_list]
    all_f_list = torch.cat(all_f_list, dim=1)
    all_crit_f_list = torch.cat(all_crit_f_list, dim=1)
    adv_list = (all_f_list-all_crit_f_list).to(agent.device)
    all_logprob_list = torch.cat(all_logprob_list, dim=1)
    all_logprob_list = all_logprob_list.unsqueeze(2).expand_as(all_f_list)
    loss = (adv_list)*all_logprob_list
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

    # # compute cosine similarity penalty
    cos_penalty = cosine_similarity(loss, ray_list.unsqueeze(1), dim=2)
    cos_penalty_per_ray = cos_penalty.mean(dim=1)
    total_loss += ld*cos_penalty_per_ray.sum()
    update_phn(phn, phn_opt, total_loss)
    agent.zero_grad(set_to_none=True)
    phn.zero_grad(set_to_none=True)
    # write_training_phn_progress(writer, loss.detach().cpu(),ray_list.cpu(),cos_penalty.detach().cpu())

def train_one_epoch(agent, phn, phn_opt, writer, batch_size, total_num_samples, num_ray, ld):
    phn.train()
    num_nodes_list = [20,30]
    num_items_per_city_list = [1,3,5]
    ic_list = [0,1,2]
    num_config = len(num_nodes_list)*len(num_items_per_city_list)*len(ic_list)
    batch_size_per_config = int(batch_size/num_config)
    num_samples = int(total_num_samples/num_config)
    max_iter = int(num_samples/batch_size_per_config)
    config_list = [(num_nodes, num_items_per_city, ic) for num_nodes in num_nodes_list for num_items_per_city in num_items_per_city_list for ic in ic_list]
    datasets = [TTPDataset(num_samples, config[0], config[1], config[2]) for config in config_list]
    dl_iter_list = [iter(DataLoader(dataset, batch_size=batch_size_per_config, shuffle=True)) for dataset in datasets]
    for i in tqdm(range(max_iter),desc="Train Epoch"):
        batch_list = [next(dl_iter) for dl_iter in dl_iter_list]
        batch_list = [combine_batch_list([batch_list[i], batch_list[i+1], batch_list[i+2]]) for i in range(0,18,3)]
        train_one_batch(agent, phn, phn_opt, batch_list, writer, num_ray, ld)

def run(args):
    agent, phn, phn_opt, last_epoch, writer, _, _ = setup_phn(args)
    vd_proc:subprocess.Popen=None
    early_stop = 0
    for epoch in range(last_epoch, args.max_epoch):
        train_one_epoch(agent, phn, phn_opt, writer, args.batch_size, args.num_training_samples, args.num_ray, args.ld)
        if vd_proc is not None:
            vd_proc.wait()
        vd = load_validator(args.title)
        if vd.is_improving:
            early_stop = 0
            save_phn(phn, phn_opt, epoch, args.title, best=True)
        else:   
            early_stop += 1
        save_phn(phn, phn_opt, epoch, args.title)
        vd_proc_cmd = ["python",
                    "validate_phn.py",
                    "--ray-hidden-size",
                    args.ray_hidden_size,
                    "--title",
                    args.title,
                    "--dataset-name",
                    args.dataset_name,
                    "--device",
                    "cpu"]
        vd_proc = subprocess.Popen(vd_proc_cmd)
        epoch += 1
    vd_proc.wait()


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    args = prepare_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)