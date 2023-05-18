import copy
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import ranksums

from policy.hv import Hypervolume
from policy.non_dominated_sorting import fast_non_dominated_sort
from setup_phn import setup_phn
from ttp.ttp_dataset import get_dataset_list
from utils import prepare_args
from utils_moo import init_phn_output
from utils_moo import compute_loss, write_training_phn_progress
from utils_moo import compute_spread_loss, generate_params, solve_one_batch, generate_rays
from utils_moo import update_phn_bp_only, save_phn, write_test_hv
    
def compute_loss_one_batch(agent, phn, critic_phn, batch, training_nondom_list, num_ray=16):
    agent.train()
    agent.gae.eval()
    ray_list = generate_rays(num_ray, phn.device, is_random=True)
    param_dict_list = generate_params(phn, ray_list)
    critic_param_dict_list = generate_params(critic_phn, ray_list)
    index_list, batch = batch
    logprob_list, f_list, sum_entropies_list = solve_one_batch(agent, param_dict_list, batch)
    with torch.no_grad():
        agent.eval()
        _, greedy_f_list, _ = solve_one_batch(agent, critic_param_dict_list, batch)
    loss_obj, cos_penalty_loss, training_nondom_list = compute_loss(logprob_list, f_list, greedy_f_list, index_list, training_nondom_list, ray_list)
    spread_loss = compute_spread_loss(logprob_list, f_list)
    
    return loss_obj, cos_penalty_loss, spread_loss, training_nondom_list
    
def train_one_epoch(args, agent, phn, phn_opt, critic_phn, training_nondom_list, writer, training_dataset_list, epoch, is_initialize=False):
    phn.train()
    batch_size_per_dataset = int(args.batch_size/len(training_dataset_list))
    training_dataloader_list = [enumerate(DataLoader(train_dataset, batch_size=batch_size_per_dataset, shuffle=True, pin_memory=True)) for train_dataset in training_dataset_list]
    is_done=False
    loss_obj_list = []
    cos_penalty_loss_list = []
    spread_loss_list = []
    if training_nondom_list is None:
        training_nondom_list = [[None for _ in range(len(training_dataset_list[i]))] for i in range(len(training_dataloader_list))]
    while not is_done:
        for i, dl_it in tqdm(enumerate(training_dataloader_list), desc="Training"):
            try:
                batch_idx, batch = next(dl_it)
                loss_obj, cos_penalty_loss, spread_loss, training_nondom_list[i] = compute_loss_one_batch(agent, phn, critic_phn, batch, training_nondom_list[i], args.num_ray)
                total_loss = loss_obj
                if is_initialize:
                    total_loss = 0
                total_loss -= 0.01*spread_loss
                total_loss += args.ld*cos_penalty_loss
                total_loss.backward()
                loss_obj_list += [loss_obj.detach().cpu().numpy()]
                cos_penalty_loss_list += [cos_penalty_loss.detach().cpu().numpy()]
                spread_loss_list += [spread_loss.detach().cpu().numpy()]
            except StopIteration:
                is_done=True
                break 
        # for name, param in phn.named_parameters():
        #     print(param.grad)
        # exit()
        update_phn_bp_only(agent, phn, phn_opt)
    loss_obj_list = np.asanyarray(loss_obj_list)
    cos_penalty_loss_list = np.asanyarray(cos_penalty_loss_list)
    spread_loss_list = np.asanyarray(spread_loss_list)
    write_training_phn_progress(writer, loss_obj_list.mean(), cos_penalty_loss_list.mean(), spread_loss_list.mean(), epoch)
    return training_nondom_list

@torch.no_grad()        
def validate_one_epoch(args, agent, phn, critic_phn, critic_solution_list, validation_nondom_list, validation_dataset_list, test_batch, test_sample_solutions, tb_writer, epoch):
    agent.eval()
    batch_size_per_dataset = int(args.batch_size/len(validation_dataset_list))
    validation_dataloader_list = [enumerate(DataLoader(validation_dataset, batch_size=batch_size_per_dataset, shuffle=False, pin_memory=True)) for validation_dataset in validation_dataset_list]
    
    #evaluate agent
    ray_list = generate_rays(args.num_ray, args.device, is_random=False)
    param_dict_list = generate_params(phn, ray_list)
    f_list = []
    is_done=False
    while not is_done:
        for dl_it in tqdm(validation_dataloader_list, desc="Validation"):
            try:
                batch_idx, batch = next(dl_it)
                index_list, batch = batch
                logprob_list, batch_f_list, sum_entropies_list = solve_one_batch(agent, param_dict_list, batch)
                f_list += [batch_f_list] 
            except StopIteration:
                is_done=True
                break
    f_list = np.concatenate(f_list,axis=1)
    f_root = "f_files"
    f_dir = pathlib.Path(".")/f_root
    model_f_dir = f_dir/args.title
    model_f_dir.mkdir(parents=True, exist_ok=True)
    f_path = model_f_dir/(args.title+"_"+str(epoch)+".pt")
    np.save(f_path.absolute(), f_list)

    #get critic solution list if not exist already
    if critic_solution_list is None:
        critic_solution_list = []
        validation_dataloader_list = [enumerate(DataLoader(validation_dataset, batch_size=batch_size_per_dataset, shuffle=False, pin_memory=True)) for validation_dataset in validation_dataset_list]
        crit_param_dict_list = generate_params(critic_phn, ray_list)
        is_done=False
        while not is_done:
            for dl_it in tqdm(validation_dataloader_list, desc="Generate Critic Solution"):
                try:
                    batch_idx, batch = next(dl_it)
                    index_list, batch = batch
                    _, crit_batch_f_list, _ = solve_one_batch(agent, crit_param_dict_list, batch)
                    critic_solution_list += [crit_batch_f_list] 
                except StopIteration:
                    is_done=True
                    break
        critic_solution_list = np.concatenate(critic_solution_list,axis=1)

    # now compare the agent's solutions hv with the critics
    # use wilcoxon signed rank
    _, num_validation_instances, _ = f_list.shape
    if validation_nondom_list is None:
        validation_nondom_list = [None for _ in range(num_validation_instances)]
    hv_list = []
    critic_hv_list = []
    for i in range(num_validation_instances):
        agent_f = f_list[:,i,:]
        critic_f = critic_solution_list[:,i,:]
        old_nondom_f = validation_nondom_list[i]
        combined_f = np.concatenate([agent_f, critic_f], axis=0)
        if old_nondom_f is not None:
            combined_f = np.concatenate([combined_f, old_nondom_f], axis=0)
        nondom_f_idx = fast_non_dominated_sort(combined_f)[0]
        nondom_f = combined_f[nondom_f_idx,:]
        validation_nondom_list[i] = nondom_f
        utopia_points = np.min(nondom_f, axis=0)
        nadir_points = np.max(nondom_f, axis=0)
        diff = nadir_points-utopia_points
        diff[diff==0] = 1
        norm_agent_f = (agent_f-utopia_points)/diff
        norm_critic_f = (critic_f-utopia_points)/diff
        agent_hv = Hypervolume(np.array([1.1,1.1])).calc(norm_agent_f)
        critic_hv = Hypervolume(np.array([1.1,1.1])).calc(norm_critic_f)
        hv_list += [agent_hv]
        critic_hv_list += [critic_hv]
    hv_list = np.asanyarray(hv_list)
    critic_hv_list = np.asanyarray(critic_hv_list)
    res = ranksums(hv_list, critic_hv_list, alternative="greater")
    is_improving=False
    if res.pvalue < 0.05:
        is_improving = True
    # if hv_list.mean() > critic_hv_list.mean():
    #     is_improving = True
    print("-----------------Validation pvalue:", res.pvalue, " ", is_improving)
    
    if is_improving:
        critic_phn.load_state_dict(copy.deepcopy(phn.state_dict()))
        critic_solution_list = f_list
    # tb_writer.add_scalar("Mean Validation HV",hv_list.mean(),epoch)
    # tb_writer.add_scalar("Std Validation HV",hv_list.std(),epoch)
    # tb_writer.add_scalar("Median Validation HV",np.median(hv_list),epoch)
    is_improving_val = 1 if is_improving else 0
    tb_writer.add_scalar("is improving?", is_improving_val, epoch)
    
    # Scatter plot with gradient colors
    ray_list = generate_rays(50, args.device, is_random=False)
    param_dict_list = generate_params(phn, ray_list)
    logprobs_list, test_f_list, sum_entropies_list = solve_one_batch(agent, param_dict_list, test_batch)
    # Define the light and dark blue colors
    light_blue = mcolors.CSS4_COLORS['lightblue']
    dark_blue = mcolors.CSS4_COLORS['darkblue']

    # Create a linear gradient from light blue to dark blue
    gradient = np.linspace(0,1,len(param_dict_list))
    colors = np.vstack((mcolors.to_rgba(light_blue), mcolors.to_rgba(dark_blue)))
    my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors, N=len(param_dict_list))

    plt.figure()
    plt.scatter(test_sample_solutions[:,0], -test_sample_solutions[:,1], c="red")
    plt.scatter(test_f_list[:,0,0], test_f_list[:,0,1], c=gradient, cmap=my_cmap)
    tb_writer.add_figure("Solutions "+args.dataset_name, plt.gcf(), epoch)
    write_test_hv(tb_writer, test_f_list[:,0,:], epoch, test_sample_solutions)
    return is_improving, critic_solution_list, validation_nondom_list


def run(args):
    patience = 100
    not_improving_count = 0
    agent, phn, phn_opt, critic_phn, critic_solution_list, training_nondom_list, validation_nondom_list, last_epoch, writer, test_batch, test_sample_solutions = setup_phn(args)
    nn_list = [10,20,30]
    nipc_list = [1,3,5]
    len_types = len(nn_list)*len(nipc_list)
    train_num_samples_per_dataset = int(args.num_training_samples/len_types)
    validation_num_samples_per_dataset = int(args.num_validation_samples/len_types)
    training_dataset_list = get_dataset_list(train_num_samples_per_dataset, nn_list, nipc_list, mode="training")
    validation_dataset_list = get_dataset_list(validation_num_samples_per_dataset, nn_list, nipc_list, mode="validation")

    
    if last_epoch == 0:
        init_phn_output(agent, phn, writer, max_step=1000)
    #     validate_one_epochv2(args, agent, phn, validator, validation_dataset,test_batch,test_sample_solutions, writer, -1)  
    #     save_phn(phn, phn_opt, -1, args.title)
        is_improving, _, validation_nondom_list = validate_one_epoch(args, agent, phn, critic_phn, critic_solution_list, validation_nondom_list, validation_dataset_list, test_batch, test_sample_solutions, writer, -1) 
    for epoch in range(last_epoch, args.max_epoch):
        if epoch <=1:
            training_nondom_list = train_one_epoch(args, agent, phn, phn_opt, critic_phn, training_nondom_list, writer, training_dataset_list, epoch, is_initialize=True)
        else:
            training_nondom_list = train_one_epoch(args, agent, phn, phn_opt, critic_phn, training_nondom_list, writer, training_dataset_list, epoch, is_initialize=False)
        is_improving, critic_solution_list, validation_nondom_list = validate_one_epoch(args, agent, phn, critic_phn, critic_solution_list, validation_nondom_list, validation_dataset_list, test_batch, test_sample_solutions, writer, epoch) 
        save_phn(phn, phn_opt, critic_phn, critic_solution_list, training_nondom_list, validation_nondom_list, epoch, args.title)
        if is_improving:
            save_phn(phn, phn_opt, critic_phn, critic_solution_list, training_nondom_list, validation_nondom_list, epoch, args.title, is_best=True)
            not_improving_count = 0
        else:
            not_improving_count += 1
        if not_improving_count == patience:
            break

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)
