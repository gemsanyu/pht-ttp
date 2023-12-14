import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from setup_phn import setup_phn
from utils import prepare_args

config_name_list = [
    "ch150_n1490_bounded-strongly-corr",
    "ch150_n1490_uncorr-similar-weights",
    "ch150_n1490_uncorr",
    "ch150_n149_bounded-strongly-corr",
    "ch150_n149_uncorr-similar-weights",
    "ch150_n149_uncorr",
    "ch150_n447_bounded-strongly-corr",
    "ch150_n447_uncorr-similar-weights",
    "ch150_n447_uncorr",
    "ch150_n745_bounded-strongly-corr",
    "ch150_n745_uncorr-similar-weights",
    "ch150_n745_uncorr",
    "eil76_n225_bounded-strongly-corr",
    "eil76_n225_uncorr-similar-weights",
    "eil76_n225_uncorr",
    "eil76_n375_bounded-strongly-corr",
    "eil76_n375_uncorr-similar-weights",
    "eil76_n375_uncorr",
    "eil76_n750_bounded-strongly-corr",
    "eil76_n750_uncorr-similar-weights",
    "eil76_n750_uncorr",
    "eil76_n75_bounded-strongly-corr",
    "eil76_n75_uncorr-similar-weights",
    "eil76_n75_uncorr",
    ]
num_instance_per_problem = 10
problems = []
for config_name in config_name_list:
    for i in range(1,num_instance_per_problem+1):
        idx = str(i)
        if i<10:
            idx = "0"+idx
        problems += [config_name+"_"+idx]

@torch.no_grad()
def test(args, agent, phn, n_solutions=200):
    agent.eval()
    phn.eval()
    
    #get static embeddings first, it can be widely reused
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    # print(ray_list)
    param_dict_list = []
    params_dm = np.zeros((n_solutions,n_solutions))
    ray_dm = np.zeros((n_solutions,n_solutions))
    for ray in ray_list:
        param_dict = phn(ray.to(agent.device))
        param_dict_list += [param_dict]
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i==j:
                continue
            ray_i = ray_list[i]
            ray_j = ray_list[j]
            ray_dist = (ray_i-ray_j).norm()
            ray_dm[i,j] = ray_dist
            p_i = param_dict_list[i]["po_weight"]
            p_j = param_dict_list[j]["po_weight"]
            p_dist = (p_j-p_i).norm()/p_i.norm()
            params_dm[i,j] = p_dist.numpy()
    ray_dm = ray_dm.flatten()
    ray_dm = np.round(ray_dm, 5)
    params_dm = params_dm.flatten()
    sort_idx = np.argsort(ray_dm)
    ray_dm = ray_dm[sort_idx]
    params_dmx = params_dm[sort_idx]
    # print(ray_dm)
    # print(params_dm)
    pm_list_dict = {}
    unique_ray_list = []
    for i in range(len(ray_dm)):
        ray = ray_dm[i]
        if str(ray) not in pm_list_dict.keys():
            unique_ray_list += [ray]
            pm_list_dict[str(ray)] = []
        pmx = params_dmx[i]
        pm_list_dict[str(ray)] += [pmx]
    unique_ray_list = np.asanyarray(unique_ray_list)
    avg_pm = []
    for ray in unique_ray_list:
        avg_pm += [np.asanyarray(pm_list_dict[str(ray)]).mean()]
    avg_pm = np.asanyarray(avg_pm)
    # plt.plot(unique_ray_list, avg_pm)
    # plt.show()
    
    folder = os.path.join(os.getcwd(), "results")
    participant = "AM-PHN"
    sort_idx = np.argsort(params_dm)
    unique_pm_list = []
    params_dmx = params_dm[sort_idx]
    params_dmx = np.round(params_dmx, 5)
    is_unique_dict = {}
    for i in range(len(params_dmx)):
            pm = params_dmx[i]
            if str(pm) not in is_unique_dict.keys():
                is_unique_dict[str(pm)] = True
                unique_pm_list += [pm]
        
    f_dm_list = []
    for pi,problem in enumerate(problems):
        # check for the corresponding file
        fname = "%s_%s.f" % (participant, problem)   
        path_to_file = os.path.join(folder,participant, fname)
        
        # in case the wrong delimiter was used
        if not os.path.isfile(path_to_file):
            fname = "%s_%s.f" % (participant, problem.replace("_", "-"))
            path_to_file = os.path.join(folder,participant, fname)
            
        # load the values in the objective space - first column is time, second profit
        _F = np.loadtxt(path_to_file)
        _F = _F * [1, -1]
        max_f = np.max(_F, axis=0, keepdims=True)
        min_f = np.min(_F, axis=0, keepdims=True)
        _F = (_F-min_f)/(max_f-min_f)
        f_dm = np.zeros((n_solutions,n_solutions))
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i==j:
                    continue
                fi = _F[i, :]
                fj = _F[j, :]
                f_dist = np.linalg.norm(fi-fj)
                f_dm[i,j] = f_dist
        f_dm = f_dm.flatten()
        f_dm = f_dm[sort_idx]

        f_list_dict = {}
        for i in range(len(params_dmx)):
            pm = params_dmx[i]
            if str(pm) not in f_list_dict.keys():
                f_list_dict[str(pm)] = []
            fd_i = f_dm[i]
            f_list_dict[str(pm)] +=[fd_i]

        avg_fdm = []
        for pm in unique_pm_list:
            # print(len(f_list_dict[str(pm)]))
            avg_fdm += [np.asanyarray(f_list_dict[str(pm)]).mean()]
        
        avg_fdm = np.asanyarray(avg_fdm)
        f_dm_list += [avg_fdm[np.newaxis,:]]

    unique_pm_list = np.asanyarray(unique_pm_list)
    f_dm_list = np.concatenate(f_dm_list)
    plt.plot(unique_pm_list, f_dm_list.mean(axis=0))
    plt.show()    
    exit()
    
    
    


def run(args):
    agent, phn, phn_opt, critic_phn, critic_solution_list, training_nondom_list, validation_nondom_list, last_epoch, writer, test_batch, test_sample_solutions = setup_phn(args, load_best=False)
    # agent.gae = agent.gae.cpu()
    test(args, agent, phn)

if __name__ == '__main__':
    args = prepare_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)