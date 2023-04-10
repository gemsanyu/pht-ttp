import math
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

def compute_loss(logprob_list, batch_f_list, greedy_batch_f_list, ray_list):
    device = logprob_list.device
    A = batch_f_list-greedy_batch_f_list
    nadir = np.max(A, axis=0, keepdims=True)
    utopia = np.min(A, axis=0, keepdims=True)
    denom = (nadir-utopia)
    denom[denom==0] = 1e-8
    norm_obj = (A-utopia)/denom
    hv_d_list = get_hv_d(A.transpose((1,0,2))).transpose(1,0)
    # compute loss now
    hv_d_list = hv_d_list.to(device)
    norm_obj = torch.from_numpy(norm_obj).to(device)
    A = torch.from_numpy(A).to(device)
    logprob_list = logprob_list.unsqueeze(2)
    loss_per_obj = logprob_list*A
    final_loss_per_obj = loss_per_obj*hv_d_list
    final_loss_per_instance = final_loss_per_obj.sum(dim=2)
    final_loss_per_ray = final_loss_per_instance.mean(dim=1)
    final_loss = final_loss_per_ray.sum()
    
    nadir = np.max(batch_f_list, axis=0, keepdims=True)
    utopia = np.min(batch_f_list, axis=0, keepdims=True)
    denom = (nadir-utopia)
    denom[denom==0] = 1e-8
    norm_obj = (batch_f_list-utopia)/denom
    
    norm_obj = torch.from_numpy(norm_obj).to(device)
    ray_list = ray_list.unsqueeze(1).expand_as(norm_obj)
    # print(logprob_list.shape, cosine_similarity(batch_f_list, ray_list, dim=2).shape)
    cos_penalty = (1-cosine_similarity(norm_obj, 1-ray_list, dim=2).unsqueeze(2))
    cos_penalty_loss = logprob_list*cos_penalty
    cos_penalty_loss_per_ray = cos_penalty_loss.mean(dim=0)
    total_cos_penalty_loss = cos_penalty_loss_per_ray.sum()
    return final_loss, total_cos_penalty_loss

def update_phn(agent, phn, opt, final_loss):
    agent.zero_grad(set_to_none=True)
    phn.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)
    final_loss.backward()
    torch.nn.utils.clip_grad_norm_(phn.parameters(), max_norm=1)
    opt.step()

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


def generate_params(phn, num_ray, device, is_random=True):
    ray_list = generate_rays(num_ray, device, is_random)
    param_dict_list = []
    for ray in ray_list:
        param_dict = phn(ray)
        param_dict_list += [param_dict]
    return ray_list, param_dict_list

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
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp = batch
    train_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp)
    static_features, dynamic_features, eligibility_mask = train_env.begin()
    num_nodes, num_items, batch_size = train_env.num_nodes, train_env.num_items, train_env.batch_size
    static_embeddings = encode(agent, static_features, num_nodes, num_items, batch_size)

    # sample rollout
    f_list, logprobs_list, sum_entropies_list = decode_one_batch(agent, param_dict_list, train_env, static_embeddings)
    return logprobs_list, f_list, sum_entropies_list

def solve_one_batchv2(agent, phn, ray_list, batch):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp = batch
    train_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp)
    static_features, dynamic_features, eligibility_mask = train_env.begin()
    num_nodes, num_items, batch_size = train_env.num_nodes, train_env.num_items, train_env.batch_size
    static_embeddings = encode(agent, static_features, num_nodes, num_items, batch_size)

    param_dict_list = generate_paramsv2(phn, ray_list, static_embeddings)

    # sample rollout
    f_list, logprobs_list, sum_entropies_list = decode_one_batch(agent, param_dict_list, train_env, static_embeddings)
    return logprobs_list, f_list, sum_entropies_list

def compute_spread_loss(logprobs, f_list, param_dict_list):
    param_list = [param_dict["v1"].ravel().unsqueeze(0) for param_dict in param_dict_list]
    param_list = torch.cat(param_list).unsqueeze(0)
    distance_matrix = torch.cdist(param_list, param_list)
    batched_distance_per_ray = torch.transpose(distance_matrix.sum(dim=2),0,1)
    spread_loss = batched_distance_per_ray.mean()
    return spread_loss

@torch.no_grad()        
def validate_one_epoch(args, agent, phn, validator, validation_dataset, test_batch, test_sample_solutions, tb_writer, epoch):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size)
    
    ray_list, param_dict_list = generate_params(phn, args.num_ray, agent.device, is_random=False)
    f_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        logprob_list, batch_f_list, sum_entropies_list = solve_one_batch(agent, param_dict_list, batch)
        f_list += [batch_f_list] 
    f_list = np.concatenate(f_list,axis=1)
    # print(f_list.shape)
    f_list = f_list.transpose((1,0,2))
    # print(f_list.shape)
    # exit()
    nadir_points = np.max(f_list, axis=1)
    utopia_points = np.min(f_list, axis=1)
    validator.insert_new_ref_points(nadir_points, utopia_points)

    nd_solutions_list = []
    for i in range(len(validation_dataset)):
        nondom_idx = fast_non_dominated_sort(f_list[i,:,:])[0]
        nd_solutions = f_list[i, nondom_idx, :]
        nd_solutions_list += [nd_solutions]
    validator.insert_new_nd_solutions(nd_solutions_list)
    validator.epoch +=1

    last_mean_running_igd = validator.get_last_mean_running_igd()
    if last_mean_running_igd is not None:
        tb_writer.add_scalar("Mean Running IGD", last_mean_running_igd, validator.epoch)
    last_mean_delta_nadir, last_mean_delta_utopia = validator.get_last_delta_refpoints()
    if last_mean_delta_nadir is not None:
        tb_writer.add_scalar("Mean Delta Nadir", last_mean_delta_nadir, validator.epoch)
        tb_writer.add_scalar("Mean Delta Utopia", last_mean_delta_utopia, validator.epoch)

    # test
    marker_list = [".","o","v","^","<",">","1","2","3","4"]
    colors_list = [key for key in mcolors.TABLEAU_COLORS.keys()]
    combination_list = [[c,m] for c in colors_list for m in marker_list]
    ray_list, param_dict_list = generate_params(phn, 50, agent.device)
    logprobs_list, test_f_list, sum_entropies_list = solve_one_batch(agent, param_dict_list, test_batch)
    plt.figure()
    for i in range(len(param_dict_list)):
        c = combination_list[i][0]
        m = combination_list[i][1]
        plt.scatter(test_f_list[i,0,0], -test_f_list[i,0,1], c=c, marker=m)
    for i in range(len(test_sample_solutions)):
        plt.scatter(test_sample_solutions[i,0], test_sample_solutions[i,1], c="red")
    tb_writer.add_figure("Solutions "+args.dataset_name, plt.gcf(), validator.epoch)

@torch.no_grad()        
def validate_one_epochv2(args, agent, phn, validator, validation_dataset, test_batch, test_sample_solutions, tb_writer, epoch):
    agent.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size)
    
    ray_list = generate_rays(args.num_ray, phn.device, is_random=False)
    # ray_list, param_dict_list = generate_params(phn, args.num_ray, agent.device, is_random=False)
    f_list = []
    for batch_idx, batch in tqdm(enumerate(validation_dataloader), desc=f'Validation epoch {epoch}'):
        logprob_list, batch_f_list, sum_entropies_list = solve_one_batchv2(agent, phn, ray_list, batch)
        f_list += [batch_f_list] 
    f_list = np.concatenate(f_list,axis=1)
    # print(f_list.shape)
    f_list = f_list.transpose((1,0,2))
    # print(f_list.shape)
    # exit()
    nadir_points = np.max(f_list, axis=1)
    utopia_points = np.min(f_list, axis=1)
    validator.insert_new_ref_points(nadir_points, utopia_points)

    nd_solutions_list = []
    for i in range(len(validation_dataset)):
        nondom_idx = fast_non_dominated_sort(f_list[i,:,:])[0]
        nd_solutions = f_list[i, nondom_idx, :]
        nd_solutions_list += [nd_solutions]
    validator.insert_new_nd_solutions(nd_solutions_list)
    validator.epoch +=1

    last_mean_running_igd = validator.get_last_mean_running_igd()
    if last_mean_running_igd is not None:
        tb_writer.add_scalar("Mean Running IGD", last_mean_running_igd, validator.epoch)
    last_mean_delta_nadir, last_mean_delta_utopia = validator.get_last_delta_refpoints()
    if last_mean_delta_nadir is not None:
        tb_writer.add_scalar("Mean Delta Nadir", last_mean_delta_nadir, validator.epoch)
        tb_writer.add_scalar("Mean Delta Utopia", last_mean_delta_utopia, validator.epoch)

    # test
    marker_list = [".","o","v","^","<",">","1","2","3","4"]
    colors_list = [key for key in mcolors.TABLEAU_COLORS.keys()]
    combination_list = [[c,m] for c in colors_list for m in marker_list]
    # ray_list, param_dict_list = generate_params(phn, 50, agent.device)
    ray_list = generate_rays(50,phn.device,is_random=False)
    logprobs_list, test_f_list, sum_entropies_list = solve_one_batchv2(agent, phn, ray_list, test_batch)
    plt.figure()
    for i in range(len(ray_list)):
        c = combination_list[i][0]
        m = combination_list[i][1]
        plt.scatter(test_f_list[i,0,0], -test_f_list[i,0,1], c=c, marker=m)
    for i in range(len(test_sample_solutions)):
        plt.scatter(test_sample_solutions[i,0], test_sample_solutions[i,1], c="red")
    tb_writer.add_figure("Solutions "+args.dataset_name, plt.gcf(), validator.epoch)
    
def write_training_phn_progress(writer, total_loss, cos_penalty):
    writer.add_scalar("Total Loss", total_loss)
    writer.add_scalar("Cos Penalty", cos_penalty)

def initialize(target_param,phn,opt,tb_writer):
    # r = random.random()
    ray = np.asanyarray([[0.5, 0.5]],dtype=float)
    ray = torch.from_numpy(ray).to(phn.device, dtype=torch.float32)
    graph_embeddings_dummy = torch.fill((1,phn.num_neurons),0.5)
    param_dict = phn(ray, graph_embeddings_dummy)
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
        elif name == "pointer.attention_layer.features_embedder.weight":
            fe1 = param.data.ravel()
        elif name == "pointer.attention_layer.query_embedder.weight":
            qe1 = param.data.ravel()
        
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

def save_phn(phn, epoch, title):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    checkpoint = {
        "phn_state_dict":phn.state_dict(),
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

