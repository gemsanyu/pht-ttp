import os
import pathlib
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.agent import Agent
from ttp.ttp_env import TTPEnv
from policy.hv import Hypervolume
from policy.non_dominated_sorting import fast_non_dominated_sort

CPU_DEVICE = torch.device('cpu')

class BatchProperty(NamedTuple):
    num_nodes: int
    num_items_per_city: int
    num_clusters: int
    item_correlation: int
    capacity_factor: int

def solve(agent: Agent, env: TTPEnv, param_dict=None):
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    static_features, node_dynamic_features, global_dynamic_features, eligibility_mask = env.begin()
    static_features = torch.from_numpy(static_features).to(agent.device)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device)
    global_dynamic_features = torch.from_numpy(global_dynamic_features).to(agent.device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
    # compute fixed static embeddings and graph embeddings once for reusage
    item_init_embed = agent.item_init_embedder(static_features[:, :env.num_items, :])
    depot_init_embed = agent.depot_init_embed.expand(size=(env.batch_size,1,-1))
    node_init_embed = agent.node_init_embed.expand(size=(env.batch_size,env.num_nodes-1,-1))
    init_embed = torch.cat([item_init_embed, depot_init_embed, node_init_embed], dim=1)
    static_embeddings, graph_embeddings = agent.gae(init_embed)
    fixed_context = agent.project_fixed_context(graph_embeddings)
    # similarly, compute glimpse_K, glimpse_V, and logits_K once for reusage
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
                                   fixed_context[is_not_finished],
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

def encode(agent:Agent, static_features, num_nodes, num_items, batch_size):
    static_features = torch.from_numpy(static_features).to(agent.device)
    item_init_embed = agent.item_init_embedder(static_features[:, :num_items, :])
    depot_init_embed = agent.depot_init_embed.expand(size=(batch_size,1,-1))
    node_init_embed = agent.node_init_embed.expand(size=(batch_size,num_nodes-1,-1))
    init_embed = torch.cat([item_init_embed, depot_init_embed, node_init_embed], dim=1)
    static_embeddings, graph_embeddings = agent.gae(init_embed)
    fixed_context = agent.project_fixed_context(graph_embeddings)
    glimpse_K_static, glimpse_V_static, logits_K_static = agent.project_embeddings(static_embeddings).chunk(3, dim=-1)
    glimpse_K_static = agent._make_heads(glimpse_K_static)
    glimpse_V_static = agent._make_heads(glimpse_V_static)
    return static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static

def solve_decode_only(agent:Agent, 
                    env:TTPEnv, 
                    static_embeddings, 
                    fixed_context,
                    glimpse_K_static, 
                    glimpse_V_static, 
                    logits_K_static,
                    param_dict=None):
    env.begin()
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    static_features, node_dynamic_features, global_dynamic_features, eligibility_mask = env.begin()
    static_features = torch.from_numpy(static_features).to(CPU_DEVICE)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(agent.device)
    global_dynamic_features = torch.from_numpy(global_dynamic_features).to(agent.device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
    
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=agent.device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    while torch.any(eligibility_mask):
        is_not_finished = torch.any(eligibility_mask, dim=1)
        active_idx = is_not_finished.nonzero().long().squeeze(1)
        previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :].unsqueeze(1)
        selected_idx, logp, entropy = agent(static_embeddings[is_not_finished],
                                   fixed_context[is_not_finished],
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

def update(agent, agent_opt, loss):
    agent_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()

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

def save_nes(policy, epoch, title, best=False):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    if best:
        checkpoint_path = checkpoint_dir/(title+"_best.pt")
        

    checkpoint = {
        "policy":policy,  
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())


def write_training_progress(tour_length, total_profit, total_cost, agent_loss, entropy_loss, critic_cost, logprob, writer):
    writer.add_scalar("Training Tour Length", tour_length)
    writer.add_scalar("Training Total Profit", total_profit)
    writer.add_scalar("Training Total Cost", total_cost)
    writer.add_scalar("Training Agent Loss", agent_loss)
    writer.add_scalar("Training Entropy Loss", entropy_loss)
    writer.add_scalar("Training NLL", -logprob)
    writer.add_scalar("Training Critic Exp Moving Average", critic_cost)
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

def write_test_phn_progress(writer, f_list, epoch, dataset_name, sample_solutions=None, nondominated_only=False):
    plt.figure()
    _f_list = f_list.clone().numpy()
    _f_list[:,1] = -_f_list[:,1]
    if sample_solutions is not None:
        _ss = sample_solutions.clone().numpy()
        _ss[:,1] = -_ss[:,1]
        _all =  np.concatenate([_f_list,_ss], axis=0)
    else:
        _all = _f_list
    _min,_max = np.min(_all, axis=0), np.max(_all, axis=0)
    _min,_max = _min[np.newaxis,:], _max[np.newaxis,:]
    _N  = (_f_list-_min)/((_max-_min)+1e-8)
    reference_point = np.array([1.1,1.1])
    hv_getter = Hypervolume(reference_point)
    total_hv = hv_getter.calc(_N)
    nondom_idx = fast_non_dominated_sort(_f_list)[0]
    if nondominated_only:
        plt.scatter(f_list[nondom_idx, 0], f_list[nondom_idx, 1], c="blue")
    else:
        plt.scatter(f_list[:, 0], f_list[:, 1], c="blue")

    if sample_solutions is not None:
        plt.scatter(sample_solutions[:, 0], sample_solutions[:, 1], c="red")
    writer.add_figure("Solutions "+dataset_name, plt.gcf(), epoch)
    writer.add_scalar("Test HV "+dataset_name, total_hv, epoch)
    writer.flush()

