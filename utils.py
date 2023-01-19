import os
import pathlib
from typing import NamedTuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from agent.agent import Agent
from policy.non_dominated_sorting import fast_non_dominated_sort
from ttp.ttp_env import TTPEnv

CPU_DEVICE = torch.device('cpu')

MASTER = 0

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

def solve(agent: Agent, env: TTPEnv, param_dict=None, normalized=False):
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    last_pointer_hidden_states = torch.zeros((agent.pointer.num_layers, env.batch_size, agent.pointer.num_neurons), device=agent.device, dtype=torch.float32)
    static_features, dynamic_features, eligibility_mask = env.begin()
    static_features =  torch.from_numpy(static_features).to(agent.device)
    item_static_embeddings = agent.item_static_encoder(static_features[:,:env.num_items,:])
    depot_static_embeddings = agent.depot_init_embed.expand((env.batch_size,1,-1))
    node_static_embeddings = agent.node_init_embed.expand((env.batch_size, env.num_nodes-1, -1))
    static_embeddings = torch.cat([item_static_embeddings, depot_static_embeddings, node_static_embeddings],dim=1)
    dynamic_features = torch.from_numpy(dynamic_features).to(agent.device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=agent.device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    # initially pakai initial input
    previous_embeddings = agent.inital_input.repeat_interleave(env.batch_size, dim=0)
    first_turn = True
    while torch.any(eligibility_mask):
        is_not_finished = torch.any(eligibility_mask, dim=1)
        active_idx = is_not_finished.nonzero().long().squeeze(1)
        if not first_turn:
            previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :]
            previous_embeddings = previous_embeddings.unsqueeze(1)
        next_pointer_hidden_states = last_pointer_hidden_states
        dynamic_embeddings = agent.dynamic_encoder(dynamic_features)
        forward_results = agent(last_pointer_hidden_states[:, active_idx, :], static_embeddings[active_idx], dynamic_embeddings[active_idx],eligibility_mask[active_idx], previous_embeddings, param_dict)
        next_pointer_hidden_states[:, active_idx, :], logits, probs = forward_results
        last_pointer_hidden_states = next_pointer_hidden_states
        selected_idx, logprob, entropy = agent.select(probs)
        #save logprobs
        logprobs[active_idx] += logprob
        sum_entropies[active_idx] += entropy
        dynamic_features, eligibility_mask = env.act(active_idx, selected_idx)
        dynamic_features = torch.from_numpy(dynamic_features).to(agent.device)
        eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
        prev_selected_idx[active_idx] = selected_idx
        first_turn = False
    # # get total profits and tour lenghts
    tour_list, item_selection, tour_lengths, total_profits, total_cost = env.finish(normalized=normalized)
    return tour_list, item_selection, tour_lengths, total_profits, total_cost, logprobs, sum_entropies

def encode(agent, static_features, num_nodes, num_items, batch_size):
    static_features =  torch.from_numpy(static_features).to(agent.device)
    item_static_embeddings = agent.item_static_encoder(static_features[:,:num_items,:])
    depot_static_embeddings = agent.depot_init_embed.expand((batch_size,1,-1))
    node_static_embeddings = agent.node_init_embed.expand((batch_size, num_nodes-1, -1))
    static_embeddings = torch.cat([item_static_embeddings, depot_static_embeddings, node_static_embeddings], dim=1)
    return static_embeddings

def solve_decode_only(agent: Agent, env: TTPEnv, static_embeddings, param_dict=None, normalized=False):
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    last_pointer_hidden_states = torch.zeros((agent.pointer.num_layers, env.batch_size, agent.pointer.num_neurons), device=agent.device, dtype=torch.float32)
    static_features, dynamic_features, eligibility_mask = env.begin()
    dynamic_features = torch.from_numpy(dynamic_features).to(agent.device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=agent.device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    # initially pakai initial input
    previous_embeddings = agent.inital_input.repeat_interleave(env.batch_size, dim=0)
    first_turn = True
    while torch.any(eligibility_mask):
        is_not_finished = torch.any(eligibility_mask, dim=1)
        active_idx = is_not_finished.nonzero().long().squeeze(1)
        if not first_turn:
            previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :]
            previous_embeddings = previous_embeddings.unsqueeze(1)
        next_pointer_hidden_states = last_pointer_hidden_states
        dynamic_embeddings = agent.dynamic_encoder(dynamic_features)
        forward_results = agent(last_pointer_hidden_states[:, active_idx, :], static_embeddings[active_idx], dynamic_embeddings[active_idx],eligibility_mask[active_idx], previous_embeddings, param_dict)
        next_pointer_hidden_states[:, active_idx, :], logits, probs = forward_results
        last_pointer_hidden_states = next_pointer_hidden_states
        selected_idx, logprob, entropy = agent.select(probs)
        #save logprobs
        logprobs[active_idx] += logprob
        sum_entropies[active_idx] += entropy
        dynamic_features, eligibility_mask = env.act(active_idx, selected_idx)
        dynamic_features = torch.from_numpy(dynamic_features).to(agent.device)
        eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
        prev_selected_idx[active_idx] = selected_idx
        first_turn = False
    # # get total profits and tour lenghts
    tour_list, item_selection, tour_lengths, total_profits, total_cost = env.finish(normalized=normalized)
    return tour_list, item_selection, tour_lengths, total_profits, total_cost, logprobs, sum_entropies

def compute_loss(total_costs, critic_costs, logprobs, sum_entropies):
    advantage = (total_costs - critic_costs).to(logprobs.device)
    # advantage = (advantage-advantage.mean())/(1e-8+advantage.std())
    agent_loss = -((advantage.detach())*logprobs).mean()
    entropy_loss = -sum_entropies.mean()
    return agent_loss, entropy_loss


def update(agent, agent_opt, loss):
    agent_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()


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

def write_test_phn_progress(writer, f_list, epoch, dataset_name, sample_solutions=None, nondominated_only=False):
    plt.figure()
    if nondominated_only:
        f_list_d = f_list.clone().numpy()
        f_list_d[:,1]=-f_list_d[:,1]
        nondom_idx = fast_non_dominated_sort(f_list_d)[0]
        plt.scatter(f_list[nondom_idx, 0], f_list[nondom_idx, 1], c="blue")
    else:
        plt.scatter(f_list[:, 0], f_list[:, 1], c="blue")
    if sample_solutions is not None:
        plt.scatter(sample_solutions[:, 0], sample_solutions[:, 1], c="red")
    writer.add_figure("Solutions "+dataset_name, plt.gcf(), epoch)
    writer.flush()


def write_training_phn_progress(mean_total_profit, mean_tour_length, profit_loss, tour_length_loss, epo_loss, logprob, num_nodes, num_items, writer):
    writer.add_scalar(f'Training PHN Mean Total Profit {num_nodes},{num_items}', mean_total_profit)
    writer.add_scalar(f'Training PHN Mean Tour Length {num_nodes},{num_items}', mean_tour_length)
    writer.add_scalar(f'Training PHN Profit Loss {num_nodes},{num_items}', profit_loss)
    writer.add_scalar(f'Training PHN Tour Length Loss {num_nodes},{num_items}', tour_length_loss)
    writer.add_scalar(f'Training PHN EPO Loss {num_nodes},{num_items}', epo_loss)
    writer.add_scalar(f'Training PHN NLL {num_nodes},{num_items}', -logprob)
    writer.flush()
