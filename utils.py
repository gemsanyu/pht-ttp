import os
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from agent.agent import Agent
from policy.normalization import normalize
from policy.hv import Hypervolume
from policy.utils import get_hv_contributions
from ttp.ttp_env import TTPEnv

CPU_DEVICE = torch.device('cpu')

class BatchProperty(NamedTuple):
    num_nodes: int
    num_items_per_city: int
    num_clusters: int
    item_correlation: int
    capacity_factor: int


def get_batch_properties(num_nodes_list, num_items_per_city_list):
    """
        training dataset information for each batch
        1 batch will represent 1 possible problem configuration
        including num of node clusters, capacity factor, item correlation
        num_nodes, num_items_per_city_list
    """
    batch_properties = []
    capacity_factor_list = [i+1 for i in range(10)]
    num_clusters_list = [1]
    item_correlation_list = [i for i in range(3)]

    for num_nodes in num_nodes_list:
        for num_items_per_city in num_items_per_city_list:
            for capacity_factor in capacity_factor_list:
                for num_clusters in num_clusters_list:
                    for item_correlation in item_correlation_list:
                        batch_property = BatchProperty(num_nodes, num_items_per_city,
                                                       num_clusters, item_correlation,
                                                       capacity_factor)
                        batch_properties += [batch_property]
    return batch_properties

def solve(agent: Agent, env: TTPEnv, param_dict=None):
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    static_features, node_dynamic_features, global_dynamic_features, eligibility_mask = env.begin()
    static_features = torch.from_numpy(static_features).to(agent.device)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device)
    global_dynamic_features = torch.from_numpy(global_dynamic_features).to(agent.device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
    # compute fixed static embeddings and graph embeddings once for reusage
    static_embeddings, graph_embeddings = agent.gae(static_features[:, :env.num_items, :])
    node_embeddings = static_embeddings[:,:env.num_items,:].unsqueeze(1).expand(env.batch_size, env.num_nodes, env.num_items, -1)
    item_city_mask = torch.from_numpy(env.item_city_mask).to(agent.device)
    item_city_mask = item_city_mask.unsqueeze(3).expand(env.batch_size, env.num_nodes, env.num_items, agent.embed_dim)
    node_embeddings = (node_embeddings*item_city_mask.float()).sum(dim=2)
    static_embeddings = torch.cat([static_embeddings, node_embeddings[:,1:,:]], dim=1)
    # similarly, compute glimpse_K, glimpse_V, and logits_K once for reusage
    # if param_dict is not None:
    #     glimpse_K_static, glimpse_V_static, logits_K_static = F.linear(static_embeddings, param_dict["pe_weight"]).chunk(3, dim=-1)
    # else:
    glimpse_K_static, glimpse_V_static, logits_K_static = agent.project_embeddings(static_embeddings).chunk(3, dim=-1)
    glimpse_K_static = agent._make_heads(glimpse_K_static)
    glimpse_V_static = agent._make_heads(glimpse_V_static)
    # glimpse_K awalnya batch_size, num_items, embed_dim
    # ubah ke batch_size, num_items, n_heads, key_dim
    # terus permute agar jadi n_heads, batch_size, num_items, key_dim
    
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=agent.device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    while torch.any(eligibility_mask):
        is_not_finished = torch.any(eligibility_mask, dim=1)
        active_idx = is_not_finished.nonzero().long().squeeze(1)
        previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :].unsqueeze(1)
        selected_idx, logp, entropy = agent(static_embeddings[is_not_finished],
                                   graph_embeddings[is_not_finished],
                                   previous_embeddings,
                                   node_dynamic_features[is_not_finished],
                                   global_dynamic_features[is_not_finished],    
                                   glimpse_V_static[:, is_not_finished, :, :],
                                   glimpse_K_static[:, is_not_finished, :, :],
                                   logits_K_static[is_not_finished],
                                   eligibility_mask[is_not_finished],
                                   param_dict)
        #save logprobs
        logprobs[is_not_finished] += logp
        sum_entropies[is_not_finished] += entropy
        node_dynamic_features, global_dynamic_features, eligibility_mask = env.act(active_idx, selected_idx)
        node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device)
        global_dynamic_features = torch.from_numpy(global_dynamic_features).to(agent.device)
        eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
        prev_selected_idx[active_idx] = selected_idx

    # get total profits and tour lenghts
    tour_list, item_selection, tour_lengths, total_profits, total_cost = env.finish()
    return tour_list, item_selection, tour_lengths, total_profits, total_cost, logprobs, sum_entropies

def solve_decode_only(agent:Agent, env:TTPEnv, static_embeddings, graph_embeddings, param_dict=None):
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    static_features, node_dynamic_features, global_dynamic_features, eligibility_mask = env.begin()
    static_features = torch.from_numpy(static_features).to(CPU_DEVICE)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device)
    global_dynamic_features = torch.from_numpy(global_dynamic_features).to(agent.device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
    
    # if param_dict is not None:
    #     glimpse_K_static, glimpse_V_static, logits_K_static = F.linear(static_embeddings, param_dict["pe_weight"]).chunk(3, dim=-1)
    # else:
    glimpse_K_static, glimpse_V_static, logits_K_static = agent.project_embeddings(static_embeddings).chunk(3, dim=-1)
    glimpse_K_static = agent._make_heads(glimpse_K_static)
    glimpse_V_static = agent._make_heads(glimpse_V_static)
    
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=agent.device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    while torch.any(eligibility_mask):
        is_not_finished = torch.any(eligibility_mask, dim=1)
        active_idx = is_not_finished.nonzero().long().squeeze(1)
        previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :].unsqueeze(1)
        selected_idx, logp, entropy = agent(static_embeddings[is_not_finished],
                                   graph_embeddings[is_not_finished],
                                   previous_embeddings,
                                   node_dynamic_features[is_not_finished],
                                   global_dynamic_features[is_not_finished],    
                                   glimpse_V_static[:, is_not_finished, :, :],
                                   glimpse_K_static[:, is_not_finished, :, :],
                                   logits_K_static[is_not_finished],
                                   eligibility_mask[is_not_finished],
                                   param_dict)
        #save logprobs
        logprobs[is_not_finished] += logp
        sum_entropies[is_not_finished] += entropy
        node_dynamic_features, global_dynamic_features, eligibility_mask = env.act(active_idx, selected_idx)
        node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device)
        global_dynamic_features = torch.from_numpy(global_dynamic_features).to(agent.device)
        eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
        prev_selected_idx[active_idx] = selected_idx

    # get total profits and tour lenghts
    tour_list, item_selection, tour_lengths, total_profits, total_cost = env.finish()
    return tour_list, item_selection, tour_lengths, total_profits, total_cost, logprobs, sum_entropies

def compute_loss(total_costs, critic_costs, logprobs, sum_entropies):
    advantage = (total_costs - critic_costs).to(logprobs.device)
    advantage = (advantage-advantage.mean())/(1e-8+advantage.std())
    agent_loss = -((advantage.detach())*logprobs).mean()
    entropy_loss = -sum_entropies.mean()
    return agent_loss, entropy_loss

def compute_multi_loss(remaining_profits, tour_lengths, logprobs):
    remaining_profits = remaining_profits.to(logprobs.device)
    tour_lengths = tour_lengths.to(logprobs.device)
    profit_loss = ((remaining_profits.float())*logprobs).mean() # maximize hence the -
    tour_length_loss = (tour_lengths*logprobs).mean()
    return profit_loss, tour_length_loss

def update(agent, agent_opt, loss):
    agent_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()

def update_phn(phn, phn_opt, loss):
    phn_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(phn.parameters(), max_norm=1)
    phn_opt.step()

def evaluate(agent, batch):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask = batch
    env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask)
    agent.eval()
    with torch.no_grad():
        tour_list, item_selection, tour_length, total_profit, total_cost, _, _ = solve(agent, env)

    return tour_list, item_selection, tour_length.item(), total_profit.item(), total_cost.item()


def save(agent: Agent, agent_opt:torch.optim.Optimizer, validation_cost, epoch, checkpoint_path):
    checkpoint = {
        "agent_state_dict":agent.state_dict(),
        "agent_opt_state_dict":agent_opt.state_dict(),  
        "validation_cost":validation_cost,
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

    # saving best checkpoint
    best_checkpoint_path = checkpoint_path.parent /(checkpoint_path.name + "_best")
    if not os.path.isfile(best_checkpoint_path.absolute()):
        torch.save(checkpoint, best_checkpoint_path)
    else:
        best_checkpoint =  torch.load(best_checkpoint_path.absolute())
        best_validation_cost = best_checkpoint["validation_cost"]
        if best_validation_cost < validation_cost:
            torch.save(checkpoint, best_checkpoint_path.absolute())

def save_phn(phn, phn_opt, epoch, checkpoint_path):
    checkpoint = {
        "phn_state_dict":phn.state_dict(),
        "phn_opt_state_dict":phn_opt.state_dict(),  
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

def write_training_progress(tour_length, total_profit, total_cost, agent_loss, entropy_loss, critic_cost, logprob, num_nodes, num_items, writer):
    env_title = " nn "+str(num_nodes)+" ni "+str(num_items)
    writer.add_scalar("Training Tour Length"+env_title, tour_length)
    writer.add_scalar("Training Total Profit"+env_title, total_profit)
    writer.add_scalar("Training Total Cost"+env_title, total_cost)
    writer.add_scalar("Training Agent Loss"+env_title, agent_loss)
    writer.add_scalar("Training Entropy Loss"+env_title, entropy_loss)
    writer.add_scalar("Training NLL"+env_title, -logprob)
    writer.add_scalar("Training Critic Exp Moving Average"+env_title, critic_cost)
    writer.flush()

def write_validation_progress(tour_length, total_profit, total_cost, logprob, writer):
    writer.add_scalar("Validation Tour Length", tour_length)
    writer.add_scalar("Validation Total Profit", total_profit)
    writer.add_scalar("Validation Total Cost", total_cost)
    writer.add_scalar("Validation NLL", -logprob)
    writer.flush()

def write_test_progress(tour_length, total_profit, total_cost, logprob, writer):
    writer.add_scalar("Test Tour Length", tour_length)
    writer.add_scalar("Test Total Profit", total_profit)
    writer.add_scalar("Test Total Cost", total_cost)
    writer.add_scalar("Test NLL", -logprob)
    writer.flush()

def write_test_phn_progress(writer, f_list, ray_list, epoch, sample_solutions=None):
    plt.figure()
    plt.scatter(f_list[:, 0], f_list[:, 1], c="blue")
    if sample_solutions is not None:
        plt.scatter(sample_solutions[:, 0], sample_solutions[:, 1], c="red")
    writer.add_figure("Solutions", plt.gcf(), epoch)

    # write the HV
    # get nadir and ideal point first
    all = torch.cat([f_list, sample_solutions]).numpy()
    ideal_point = np.min(all, axis=0)
    nadir_point = np.max(all, axis=0)
    _N = normalize(f_list.numpy(), ideal_point, nadir_point)
    _hv = Hypervolume(np.array([1,1])).calc(_N)
    writer.add_scalar('Test HV', _hv)
    _N = torch.from_numpy(_N)
    # write hv contribution per ray
    hv_contributions = get_hv_contributions(_N)
    hv_contribution_dict = {}
    for i,ray in enumerate(ray_list):
        hv_contribution_dict["ray-"+str(i)]=hv_contributions[i]
    writer.add_scalars("Test HV Contribution", hv_contribution_dict)

    # write penalty total and per solutions
    cos_penalty = F.cosine_similarity(_N, ray_list, dim=1)
    cos_penalty_dict ={}
    for i,ray in enumerate(ray_list):
        cos_penalty_dict["ray-"+str(i)]=cos_penalty[i]
    writer.add_scalars("Test Cos Penalty", cos_penalty_dict)
    writer.add_scalar("Test Total Cos Penalty", cos_penalty.sum())
    writer.flush()

def write_training_phn_progress(writer, f_list, ray_list, cos_penalty_list):
    
    num_sol, batch_size, _ = f_list.shape
    # write the HV
    _N = f_list.numpy()
    mean_hv = 0
    for i in range(batch_size):
        _hv = Hypervolume(np.array([1,1])).calc(_N[:, i, :])
        mean_hv += _hv
    mean_hv /= batch_size
    writer.add_scalar('Train HV', mean_hv)
    # write hv contribution per ray
    mean_hvc = None
    for i in range(batch_size):
        hv_contributions = get_hv_contributions(f_list[:, i, :])
        if mean_hvc is None:
            mean_hvc = hv_contributions
        else:
            mean_hvc += hv_contributions
    mean_hvc/= batch_size
    hv_contribution_dict = {}
    for i,ray in enumerate(ray_list):
        hv_contribution_dict["ray-"+str(i)]=mean_hvc[i]
    writer.add_scalars("Train HV Contribution", hv_contribution_dict)

    # write penalty total and per solutions
    cos_penalty_ray = cos_penalty_list.sum(dim=1)
    cos_penalty_dict ={}
    for i,ray in enumerate(ray_list):
        cos_penalty_dict["ray-"+str(i)]=cos_penalty_ray[i]
    writer.add_scalars("Train Cos Penalty", cos_penalty_dict)
    writer.add_scalar("Train  Total Cos Penalty", cos_penalty_list.sum())
    writer.flush()