import os
import pathlib
import sys

import torch

from arguments import get_parser
from agent.agent import Agent
from ttp.ttp_env import TTPEnv

CPU_DEVICE = torch.device('cpu')

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

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
    tour_list, item_selection, tour_lengths, total_profits, travel_cost, total_cost = env.finish()
    return tour_list, item_selection, tour_lengths, total_profits, travel_cost, total_cost, logprobs, sum_entropies


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
    tour_list, item_selection, tour_lengths, total_profits, travel_cost, total_cost = env.finish()
    return tour_list, item_selection, tour_lengths, total_profits, travel_cost, total_cost, logprobs, sum_entropies


def compute_loss(total_costs, critic_costs, logprobs, sum_entropies):
    advantage = (total_costs - critic_costs).to(logprobs.device)
    advantage = (advantage-advantage.mean())/(1e-8+advantage.std())
    agent_loss = -((advantage.detach())*logprobs).mean()
    entropy_loss = -sum_entropies.mean()
    return agent_loss, entropy_loss

def compute_multi_loss(total_profits, travel_costs, critic_total_profits, critic_travel_costs, ray, logprobs, sum_entropies):
    ws_cost = ray[0]*total_profits - ray[1]*travel_costs
    crit_ws_cost = ray[0]*critic_total_profits - ray[1]*critic_travel_costs
    # profit_adv = critic_total_profits-total_profits
    # travel_adv = travel_costs-critic_travel_costs
    advantage = crit_ws_cost-ws_cost
    advantage = torch.from_numpy(advantage).to(logprobs.device)
    agent_loss = (advantage*logprobs).mean()
    entropy_loss = -sum_entropies.mean()
    return agent_loss, entropy_loss

def update(agent, agent_opt, loss):
    agent_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()

    
def update_bp_only(agent, agent_opt):
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()
    agent_opt.zero_grad(set_to_none=True)

def save(agent, agent_opt, critic, crit_ws_cost_list, title, weight_idx, total_weight, epoch, is_best=False):
    agent_title = title + str(weight_idx) + "_" + str(total_weight)
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/agent_title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    if is_best:
        checkpoint_path = checkpoint_dir/(title+".pt_best")
    
    checkpoint = {
        "agent_state_dict":agent.state_dict(),
        "critic_state_dict":critic.state_dict(),
        "crit_ws_cost_list": crit_ws_cost_list,
        "agent_opt_state_dict":agent_opt.state_dict(),
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

def write_training_progress(tour_length, total_profit, travel_cost, total_cost, agent_loss, entropy_loss, logprobs, sum_entropies, epoch, writer):
    # env_title = " nn "+str(num_nodes)+" ni "+str(num_items)
    writer.add_scalar("Training Average Tour Length", tour_length, epoch)
    writer.add_scalar("Training Average Total Profit", total_profit, epoch)
    writer.add_scalar("Training Average Travel Cost", travel_cost, epoch)
    writer.add_scalar("Training Average Total Cost", total_cost, epoch)
    writer.add_scalar("Training Average Agent Loss", agent_loss, epoch)
    writer.add_scalar("Training Average Entropy Loss", entropy_loss, epoch)
    writer.add_scalar("Training NLL", -logprobs, epoch)
    writer.add_scalar("Training Average Sum of Entropies", sum_entropies, epoch)
    writer.flush()

def write_validation_progress(tour_length, total_profit, travel_cost, total_cost, mean_entropies, logprob, epoch, writer):
    writer.add_scalar("Validation Tour Length", tour_length, epoch)
    writer.add_scalar("Validation Total Profit", total_profit, epoch)
    writer.add_scalar("Validation Travel Cost", travel_cost, epoch)
    writer.add_scalar("Validation Total Cost", total_cost, epoch)
    writer.add_scalar("Validation Entropies", mean_entropies, epoch)
    writer.add_scalar("Validation NLL", -logprob, epoch)
    writer.flush()

def write_test_progress(tour_length, total_profit, travel_cost, total_cost, logprob, writer):
    writer.add_scalar("Test Tour Length", tour_length)
    writer.add_scalar("Test Total Profit", total_profit)
    writer.add_scalar("Test Travel Cost", travel_cost)
    writer.add_scalar("Test Total Cost", total_cost)
    writer.add_scalar("Test NLL", -logprob)
    writer.flush()
