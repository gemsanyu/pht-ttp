import os
import pathlib
import sys
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from agent.agent import Agent, make_heads
from agent.encoder import Encoder
from arguments import get_parser
from policy.normalization import normalize
from policy.non_dominated_sorting import fast_non_dominated_sort
from policy.hv import Hypervolume
from policy.utils import get_hv_contributions
from ttp.ttp_env import TTPEnv

CPU_DEVICE = torch.device('cpu')

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args



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
    node_init_embed = agent.node_init_embed(static_features[:,env.num_items+1:,:])
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
        selected_idx, logp, entropy = agent(env.num_items,
                                   static_embeddings[is_not_finished],
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

def solve_decode_only(agent:Agent, 
                    env:TTPEnv, 
                    static_embeddings, 
                    fixed_context,
                    glimpse_K_static, 
                    glimpse_V_static, 
                    logits_K_static,
                    param_dict=None):
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
        selected_idx, logp, entropy = agent(env.num_items,
                                   static_embeddings[is_not_finished],
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

def solve_decode_only_inf(agent:Agent,
                    device, 
                    env:TTPEnv, 
                    static_embeddings, 
                    fixed_context,
                    glimpse_K_static, 
                    glimpse_V_static, 
                    logits_K_static,
                    param_dict=None):
    logprobs = torch.zeros((env.batch_size,), device=device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=device, dtype=torch.float32)
    static_features, node_dynamic_features, global_dynamic_features, eligibility_mask = env.begin()
    static_features = torch.from_numpy(static_features).to(CPU_DEVICE)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(device)
    global_dynamic_features = torch.from_numpy(global_dynamic_features).to(device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(device)
    active_idx = torch.zeros((1,), dtype=int)
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    num_items = torch.tensor(env.num_items).to(device)
    while torch.any(eligibility_mask):
        previous_embeddings = static_embeddings[:, prev_selected_idx, :]
        selected_idx, logp, entropy = agent(num_items,
                                   static_embeddings,
                                   fixed_context,
                                   previous_embeddings,
                                   node_dynamic_features,
                                   global_dynamic_features,    
                                   glimpse_V_static,
                                   glimpse_K_static,
                                   logits_K_static,
                                   eligibility_mask,
                                   param_dict)
        #save logprobs
        logprobs += logp
        sum_entropies += entropy
        node_dynamic_features, global_dynamic_features, eligibility_mask = env.act(active_idx, selected_idx)
        node_dynamic_features = torch.from_numpy(node_dynamic_features).to(device)
        global_dynamic_features = torch.from_numpy(global_dynamic_features).to(device)
        eligibility_mask = torch.from_numpy(eligibility_mask).to(device)
        prev_selected_idx = selected_idx

    # get total profits and tour lenghts
    tour_list, item_selection, tour_lengths, total_profits, total_cost = env.finish()
    return tour_list, item_selection, tour_lengths, total_profits, total_cost, logprobs, sum_entropies

def standardize(A):
    return (A-A.mean())/(1e-8+A.std())

def compute_single_loss(total_costs, critic_total_costs, logprobs, sum_entropies):
    device = logprobs.device
    # profit_adv = (critic_total_profits-total_profits) # want to increase
    # tour_length_adv = (tour_lengths-critic_tour_lengths) # want to decrease
    # profit_adv = standardize(profit_adv)
    # tour_length_adv = standardize(tour_length_adv)
    # profit_adv = torch.from_numpy(profit_adv).to(device)
    # tour_length_adv = torch.from_numpy(tour_length_adv).to(device)
    # profit_loss = (profit_adv*logprobs).mean()
    # tour_length_loss = (tour_length_adv*logprobs).mean()
    # agent_loss = 0.5*(profit_loss+tour_length_loss)
    # just the same as:
    # agent_loss = (logprobs*(0*profit_adv+1*tour_length_adv)).mean()
    adv = critic_total_costs-total_costs # want to minimize
    # adv = normalize(adv)
    agent_loss = (logprobs*torch.from_numpy(adv).to(device)).mean()
    entropy_loss = -sum_entropies.mean()
    return agent_loss, entropy_loss, adv

def update(agent, agent_opt, loss):
    agent_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()

    
def update_bp_only(agent, agent_opt):
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()
    agent_opt.zero_grad(set_to_none=True)
    


def save(agent: Agent, agent_opt:torch.optim.Optimizer, critic: Agent, critic_total_cost_list, title, epoch, is_best=False):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    if is_best:
        checkpoint_path = checkpoint_dir/(title+".pt_best")
    checkpoint = {
        "agent_state_dict":agent.state_dict(),
        "agent_opt_state_dict":agent_opt.state_dict(),  
        "critic_state_dict":critic.state_dict(),
        "critic_total_cost_list":critic_total_cost_list,
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

def encode(encoder: Encoder, static_features, num_nodes, num_items, batch_size):
    static_features = torch.from_numpy(static_features).to(encoder.device)
    item_init_embed = encoder.item_init_embedder(static_features[:, :num_items, :])
    depot_init_embed = encoder.depot_init_embed.expand(size=(batch_size,1,-1))
    node_init_embed = encoder.node_init_embedder(static_features[:,num_items+1:,:])
    init_embed = torch.cat([item_init_embed, depot_init_embed, node_init_embed], dim=1)
    static_embeddings, graph_embeddings = encoder.gae(init_embed)
    fixed_context = encoder.project_fixed_context(graph_embeddings)
    glimpse_K_static, glimpse_V_static, logits_K_static = encoder.project_embeddings(static_embeddings).chunk(3, dim=-1)
    glimpse_K_static = make_heads(glimpse_K_static)
    glimpse_V_static = make_heads(glimpse_V_static)
    return static_embeddings, fixed_context, glimpse_K_static, glimpse_V_static, logits_K_static


def write_training_progress(tour_length, total_profit, total_cost, agent_loss, entropy_loss, logprobs, sum_entropies, epoch, writer):
    # env_title = " nn "+str(num_nodes)+" ni "+str(num_items)
    writer.add_scalar("Training Average Tour Length", tour_length, epoch)
    writer.add_scalar("Training Average Total Profit", total_profit, epoch)
    writer.add_scalar("Training Average Total Cost", total_cost, epoch)
    writer.add_scalar("Training Average Agent Loss", agent_loss, epoch)
    writer.add_scalar("Training Average Entropy Loss", entropy_loss, epoch)
    writer.add_scalar("Training NLL", -logprobs, epoch)
    writer.add_scalar("Training Average Sum of Entropies", sum_entropies, epoch)
    writer.flush()

def write_validation_progress(tour_length, total_profit, total_cost, mean_entropies, logprob, epoch, writer):
    writer.add_scalar("Validation Tour Length", tour_length, epoch)
    writer.add_scalar("Validation Total Profit", total_profit, epoch)
    writer.add_scalar("Validation Total Cost", total_cost, epoch)
    writer.add_scalar("Validation Entropies", mean_entropies, epoch)
    writer.add_scalar("Validation NLL", -logprob, epoch)
    writer.flush()

def write_test_progress(tour_length, total_profit, total_cost, logprob, writer):
    writer.add_scalar("Test Tour Length", tour_length)
    writer.add_scalar("Test Total Profit", total_profit)
    writer.add_scalar("Test Total Cost", total_cost)
    writer.add_scalar("Test NLL", -logprob)
    writer.flush()