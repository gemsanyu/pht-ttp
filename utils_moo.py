import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from arguments import get_parser
from arguments import get_parser
from ttp.ttp_env import TTPEnv
from policy.non_dominated_sorting import fast_non_dominated_sort
from utils import encode, solve_decode_only
from solver.hv_maximization import HvMaximization
from policy.normalization import normalize
from policy.hv import Hypervolume

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def get_hv_d(batch_f_list):
    hv_d_list = [] 
    batch_size, num_sample, num_obj = batch_f_list.shape
    mo_opt = HvMaximization(n_mo_sol=num_sample, n_mo_obj=num_obj)
    for i in range(batch_size):
        obj_instance = np.transpose(batch_f_list[i,:,:])
        hv_d = mo_opt.compute_weights(obj_instance).transpose(0,1)
        hv_d_list += [hv_d.unsqueeze(0)]
    hv_d_list = torch.cat(hv_d_list, dim=0)
    return hv_d_list

def compute_loss(logprob_list, batch_f_list, greedy_batch_f_list, index_list, training_nondom_list, ray_list):
    device = logprob_list.device
    # combined_f = np.concatenate([batch_f_list, greedy_batch_f_list], axis=0)
    _,num_instances,_ = batch_f_list.shape
    nadir = []
    utopia= []
    for i in range(num_instances):
        old_nondom_f = training_nondom_list[index_list[i]]
        f_list = batch_f_list[:,i,:]
        crit_f_list = greedy_batch_f_list[:,i,:]
        combined_f = np.concatenate([f_list, crit_f_list])
        if old_nondom_f is not None:
            combined_f = np.concatenate([combined_f, old_nondom_f])
        nondom_f_idx = fast_non_dominated_sort(combined_f)[0]
        nondom_f = combined_f[nondom_f_idx, :]
        max_f = np.max(nondom_f, axis=0, keepdims=True)
        min_f = np.min(nondom_f, axis=0, keepdims=True)
        nadir += [max_f]
        utopia += [min_f]
        training_nondom_list[index_list[i]] = nondom_f
    nadir = np.concatenate(nadir, axis=0)
    nadir = nadir[np.newaxis,:,:]
    utopia = np.concatenate(utopia, axis=0)
    utopia = utopia[np.newaxis,:,:]
    # print(utopia.shape, nadir.shape)
    # print(utopia, nadir)
    # exit()
    # nadir = np.max(combined_f, axis=0, keepdims=True)
    # utopia = np.min(combined_f, axis=0, keepdims=True)
    denom = (nadir-utopia)
    denom[denom==0] = 1
    batch_f_list = (batch_f_list-utopia)/denom
    greedy_batch_f_list = (greedy_batch_f_list-utopia)/denom
    # norm_obj = (A-utopia)/denom
    A = batch_f_list-greedy_batch_f_list
    
    hv_d_list = get_hv_d(A.transpose((1,0,2))).transpose(1,0)
    # compute loss now
    hv_d_list = hv_d_list.to(device)
    # norm_obj = torch.from_numpy(norm_obj).to(device)
    A = torch.from_numpy(A).to(device)
    logprob_list = logprob_list.unsqueeze(2)
    loss_per_obj = logprob_list*A
    final_loss_per_obj = loss_per_obj*hv_d_list
    final_loss_per_instance = final_loss_per_obj.sum(dim=2)
    final_loss_per_ray = final_loss_per_instance.mean(dim=1)
    final_loss = final_loss_per_ray.sum()
    
    # nadir = np.max(batch_f_list, axis=0, keepdims=True)
    # utopia = np.min(batch_f_list, axis=0, keepdims=True)
    # denom = (nadir-utopia)
    # denom[denom==0] = 1            
    # norm_obj = (batch_f_list-utopia)/denom
    
    # norm_obj = torch.from_numpy(norm_obj).to(device)
    batch_f_list = torch.from_numpy(batch_f_list).to(device)
    greedy_batch_f_list = torch.from_numpy(greedy_batch_f_list).to(device)
    ray_list = ray_list.unsqueeze(1).expand_as(batch_f_list)
    # print(logprob_list.shape, cosine_similarity(batch_f_list, ray_list, dim=2).shape)
    # cos_penalty = (1-cosine_similarity(norm_obj, ray_list, dim=2).unsqueeze(2))
    cos_penalty = cosine_similarity(batch_f_list, ray_list, dim=2).unsqueeze(2)
    critic_cos_penalty = cosine_similarity(greedy_batch_f_list, ray_list, dim=2).unsqueeze(2)
    A_cos = critic_cos_penalty - cos_penalty
    cos_penalty_loss = logprob_list*A_cos
    cos_penalty_loss_per_ray = cos_penalty_loss.mean(dim=0)
    total_cos_penalty_loss = cos_penalty_loss_per_ray.sum()
    return final_loss, total_cos_penalty_loss, training_nondom_list

def update_phn(agent, phn, opt, final_loss):
    agent.zero_grad(set_to_none=True)
    phn.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)
    final_loss.backward()
    torch.nn.utils.clip_grad_norm_(phn.parameters(), max_norm=1)
    opt.step()

def update_phn_bp_only(agent, phn, opt):
    torch.nn.utils.clip_grad_norm_(phn.parameters(), max_norm=1)
    opt.step()
    agent.zero_grad(set_to_none=True)
    phn.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)
    
def generate_rays(num_ray, device, is_random=True):
    ray_list = []
    if is_random:
        start, end = 0.1, np.pi/2-0.1
        for i in range(num_ray):
            r = np.random.uniform(start + i*(end-start)/num_ray, start+ (i+1)*(end-start)/num_ray)
            ray = np.array([np.cos(r),np.sin(r)], dtype='float32')
            ray /= ray.sum()
            ray *= np.random.randint(1, 5)*abs(np.random.normal(1, 0.2))
            ray = torch.from_numpy(ray).to(device)
            ray_list += [ray]
        ray_list = torch.stack(ray_list)
    else:
        for i in range(num_ray):
            z = i/(num_ray-1)
            ray_list += [[z,1-z]]
        ray_list = torch.tensor(ray_list, dtype=torch.float32, device=device)
    return ray_list


def generate_params_and_rays(phn, num_ray, device, is_random=True):
    ray_list = generate_rays(num_ray, device, is_random)
    param_dict_list = []
    for ray in ray_list:
        param_dict = phn(ray)
        param_dict_list += [param_dict]
    return ray_list, param_dict_list

def generate_params(phn, ray_list):
    param_dict_list = []
    for ray in ray_list:
        param_dict = phn(ray)
        param_dict_list += [param_dict]
    return param_dict_list

def decode_one_batch(agent, param_dict_list, train_env, static_embeddings):
    pop_size = len(param_dict_list)
    batch_size = train_env.batch_size
    travel_time_list = np.zeros((pop_size, batch_size), dtype=np.float32)
    total_profit_list = np.zeros((pop_size, batch_size), dtype=np.float32)
    logprobs_list = torch.zeros((pop_size, batch_size), dtype=torch.float32, device=agent.device)
    sum_entropies_list = torch.zeros((pop_size, batch_size), dtype=torch.float32, device=agent.device)
    for n, param_dict in enumerate(param_dict_list):
        solve_output = solve_decode_only(agent, train_env, static_embeddings, param_dict)
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_output
        travel_time_list[n,:] = tour_lengths
        total_profit_list[n,:] = total_profits
        logprobs_list[n,:] = logprobs
        sum_entropies_list[n,:] = sum_entropies
    inv_total_profit_list = -total_profit_list 
    f_list = np.concatenate((travel_time_list[:,:,np.newaxis],inv_total_profit_list[:,:,np.newaxis]), axis=-1)
    return f_list, logprobs_list, sum_entropies_list


def generate_paramsv2(phn, ray_list, static_embeddings):
    param_dict_list = []
    graph_embeddings = static_embeddings.mean(dim=1)
    for ray in ray_list:
        param_dict = phn(ray, graph_embeddings)
        param_dict_list += [param_dict]
    return param_dict_list

def solve_one_batch(agent, param_dict_list, batch):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
    train_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask,  best_profit_kp, best_route_length_tsp)
    static_features, dynamic_features, eligibility_mask = train_env.begin()
    num_nodes, num_items, batch_size = train_env.num_nodes, train_env.num_items, train_env.batch_size
    static_embeddings = encode(agent, static_features, num_nodes, num_items, batch_size)

    # sample rollout
    f_list, logprobs_list, sum_entropies_list = decode_one_batch(agent, param_dict_list, train_env, static_embeddings)
    return logprobs_list, f_list, sum_entropies_list

def compute_spread_loss(logprobs, f_list):
    # param_list = [param_dict["v1"].ravel().unsqueeze(0) for param_dict in param_dict_list]
    # param_list = torch.cat(param_list).unsqueeze(0)
    f_list = torch.from_numpy(f_list)
    distance_matrix = torch.cdist(f_list.transpose(0,1), f_list.transpose(0,1))
    batched_distance_per_ray = torch.transpose(distance_matrix.sum(dim=2),0,1)
    batched_distance_per_ray = batched_distance_per_ray.to(logprobs.device)
    spread_loss = (logprobs*batched_distance_per_ray).mean()
    return spread_loss

def write_test_hv(writer, f_list, epoch, sample_solutions=None):
    # write the HV
    # get nadir and ideal point first
    all = np.concatenate([f_list, sample_solutions])
    ideal_point = np.min(all, axis=0)
    nadir_point = np.max(all, axis=0)
    _N = normalize(f_list, ideal_point, nadir_point)
    _hv = Hypervolume(np.array([1,1])).calc(_N)
    writer.add_scalar('Test HV', _hv, epoch)
    writer.flush()


def write_training_phn_progress(writer, loss_obj, cos_penalty_loss):
    writer.add_scalar("HV Loss", loss_obj)
    writer.add_scalar("Cos Penalty Loss", cos_penalty_loss)

def initialize(target_param,phn,opt,tb_writer):
    # r = random.random()
    ray = np.asanyarray([[0.5, 0.5]],dtype=float)
    ray = torch.from_numpy(ray).to(phn.device, dtype=torch.float32)
    param_dict = phn(ray)
    v = (param_dict["v1"]).ravel()
    # fe1 = (param_dict["fe1_weight"]).ravel()
    # qe1 = (param_dict["qe1_weight"]).ravel()
    param = torch.cat([v])
    # param=v
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
        # elif name == "pointer.attention_layer.query_embedder.weight":
        #     qe1 = param.data.ravel()
        
    v = v.detach().clone()
    # fe1 = fe1.detach().clone()
    # qe1 = qe1.detach().clone()
    target_param = torch.cat([v])
    # target_param=v
    opt_init = torch.optim.Adam(phn.parameters(), lr=1e-4)
    for i in range(max_step):
        loss = initialize(target_param,phn,opt_init,tb_writer)
        if loss < 1e-3:
            break

def save_phn(phn, phn_opt, critic_phn, critic_solution_list, training_nondom_list, validation_nondom_list, epoch, title, is_best=False):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    if is_best:
        checkpoint_path = checkpoint_dir/(title+".pt_best")
    checkpoint = {
        "phn_state_dict":phn.state_dict(),
        "phn_opt_state_dict":phn_opt.state_dict(),
        "critic_phn_state_dict":critic_phn.state_dict(),
        "critic_solution_list":critic_solution_list,
        "training_nondom_list":training_nondom_list, 
        "validation_nondom_list": validation_nondom_list,
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

