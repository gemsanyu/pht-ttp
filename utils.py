import os
from typing import NamedTuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from scipy.stats import ttest_rel

from agent.agent import Agent
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

def solve(node_agent: Agent, item_agent: Agent, env: TTPEnv, param_dict=None, normalized=False):
    logprobs = torch.zeros((env.batch_size,), device=node_agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=node_agent.device, dtype=torch.float32)
    # last_node_pointer_hidden_states = torch.zeros((node_agent.pointer.num_layers, env.batch_size, node_agent.pointer.num_neurons), device=node_agent.device, dtype=torch.float32)
    # last_item_pointer_hidden_states = torch.zeros((node_agent.pointer.num_layers, env.batch_size, node_agent.pointer.num_neurons), device=node_agent.device, dtype=torch.float32)
    last_pointer_hidden_states = torch.zeros((node_agent.pointer.num_layers, env.batch_size, node_agent.pointer.num_neurons), device=node_agent.device, dtype=torch.float32)

    env.reset()
    state = env.get_current_state()
    # embed static features once
    raw_item_static_features = torch.from_numpy(env.raw_item_static_features).to(item_agent.device)
    raw_node_static_features = torch.from_numpy(env.raw_node_static_features).to(node_agent.device)
    item_static_embeddings = item_agent.static_encoder(raw_item_static_features)
    node_static_embeddings = node_agent.static_encoder(raw_node_static_features)
    is_finished = False
    while not is_finished:
        # take item and continue
        if state.item_state is not None:
            item_state = state.item_state
            active_idx = item_state.active_idx
            raw_dynamic_features = torch.from_numpy(item_state.raw_dynamic_features).to(item_agent.device)
            dynamic_embeddings = item_agent.dynamic_encoder(raw_dynamic_features)
            eligibility_mask = torch.from_numpy(item_state.eligibility_mask).to(item_agent.device)
            prev_selected_item_idx = torch.from_numpy(item_state.prev_selected_idx).to(item_agent.device)
            # select active static embeddings
            active_current_location = env.current_location[active_idx]
            active_current_item_city_mask = torch.from_numpy(env.item_city_mask[active_idx, active_current_location])
            active_item_static_embeddings = item_static_embeddings[active_idx][active_current_item_city_mask].reshape((len(active_idx), env.num_items_per_city, -1))
            # add dummy item (aka stop symbol)
            dummy_eligibility = torch.ones((len(active_idx), 1), dtype=torch.bool, device=item_agent.device)
            dummy_embeddings = torch.zeros((len(active_idx), 1, item_agent.embedding_size), dtype=torch.float32, device=item_agent.device)
            eligibility_mask = torch.cat((eligibility_mask, dummy_eligibility), dim=1)
            dynamic_embeddings = torch.cat((dynamic_embeddings, dummy_embeddings), dim=1)
            active_item_static_embeddings = torch.cat((active_item_static_embeddings, dummy_embeddings), dim=1)
            # get previous items embeddings
            is_first_selection = (prev_selected_item_idx == -1)
            previous_item_embeddings = item_static_embeddings[active_idx, prev_selected_item_idx, :]
            previous_item_embeddings = previous_item_embeddings.unsqueeze(1)
            previous_item_embeddings[is_first_selection] = item_agent.inital_input
            forward_results = item_agent(last_pointer_hidden_states[:, active_idx, :], active_item_static_embeddings, dynamic_embeddings, eligibility_mask, previous_item_embeddings)
            last_pointer_hidden_states[:, active_idx, :], logits, probs = forward_results
            selected_item_with_dummy_idx, logprob, entropy = item_agent.select(probs)
            logprobs[active_idx] += logprob
            sum_entropies[active_idx] += entropy
            env.take_item_with_dummy(active_idx, selected_item_with_dummy_idx)
            state = env.get_current_state()
            is_finished = (state.item_state is None) and (state.node_state is None) 
            continue
        # visit nodes
        node_state = state.node_state
        active_idx = node_state.active_idx
        raw_dynamic_features = torch.from_numpy(node_state.raw_dynamic_features).to(node_agent.device)
        dynamic_embeddings = node_agent.dynamic_encoder(raw_dynamic_features)
        eligibility_mask = torch.from_numpy(node_state.eligibility_mask).to(node_agent.device)
        prev_selected_node_idx = torch.from_numpy(node_state.prev_selected_idx).to(node_agent.device)
        # idx == -1, then first selection, use learned initial embeddings  
        is_first_selection = (prev_selected_node_idx == -1)
        previous_node_embeddings = node_static_embeddings[active_idx, prev_selected_node_idx, :]
        previous_node_embeddings = previous_node_embeddings.unsqueeze(1)
        previous_node_embeddings[is_first_selection] = node_agent.inital_input
        forward_results = node_agent(last_pointer_hidden_states[:, active_idx, :], node_static_embeddings[active_idx], dynamic_embeddings, eligibility_mask, previous_node_embeddings)
        last_pointer_hidden_states[:, active_idx, :], logits, probs = forward_results
        selected_node_idx, logprob, entropy = node_agent.select(probs)
        logprobs[active_idx] += logprob
        sum_entropies[active_idx] += entropy
        env.visit_node(active_idx, selected_node_idx)
        state = env.get_current_state()
        is_finished = (state.item_state is None) and (state.node_state is None)
    
    tour_list, item_selection, tour_lengths, total_profits, total_cost = env.finish(normalized=normalized)
    return tour_list, item_selection, tour_lengths, total_profits, total_cost, logprobs, sum_entropies

def compute_loss(tour_lengths, best_tour_lengths, total_profits, best_profits, logprobs, sum_entropies):
    # cost_advantage = (total_costs - critic_costs).to(logprobs.device)
    # cost_advantage = (cost_advantage-cost_advantage.mean())/(1e-8+cost_advantage.std())
    tour_adv = (best_tour_lengths-tour_lengths).to(logprobs.device)
    tour_adv = (tour_adv-tour_adv.mean())/(1e-8+tour_adv.std())
    profit_adv = (total_profits-best_profits).to(logprobs.device)
    profit_adv = (profit_adv-profit_adv.mean())/(1e-8+profit_adv.mean())

    # cost_loss = -((cost_advantage.detach())*logprobs).mean()
    # agent_loss = cost_loss
    agent_loss = -((tour_adv*0.5+profit_adv*0.5)*logprobs).mean()
    entropy_loss = -sum_entropies.mean()
    return agent_loss, entropy_loss

def compute_multi_loss(remaining_profits, tour_lengths, logprobs):
    remaining_profits = remaining_profits.to(logprobs.device)
    tour_lengths = tour_lengths.to(logprobs.device)
    profit_loss = ((remaining_profits.float())*logprobs).mean() # maximize hence the -
    tour_length_loss = (tour_lengths*logprobs).mean()
    return profit_loss, tour_length_loss

def update(node_agent, item_agent, agent_opt, loss):
    agent_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(node_agent.parameters(), max_norm=1)
    torch.nn.utils.clip_grad_norm_(item_agent.parameters(), max_norm=1)
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

def write_test_phn_progress(writer, f_list, epoch):
    plt.figure()
    plt.scatter(f_list[:, 0], f_list[:, 1], c="blue")
    # plt.scatter(
    #     problem.sample_solutions[:, 0], problem.sample_solutions[:, 1], c="red")
    writer.add_figure("Solutions", plt.gcf(), epoch)
    writer.flush()


def write_training_phn_progress(mean_total_profit, mean_tour_length, profit_loss, tour_length_loss, epo_loss, logprob, num_nodes, num_items, writer):
    writer.add_scalar(f'Training PHN Mean Total Profit {num_nodes},{num_items}', mean_total_profit)
    writer.add_scalar(f'Training PHN Mean Tour Length {num_nodes},{num_items}', mean_tour_length)
    writer.add_scalar(f'Training PHN Profit Loss {num_nodes},{num_items}', profit_loss)
    writer.add_scalar(f'Training PHN Tour Length Loss {num_nodes},{num_items}', tour_length_loss)
    writer.add_scalar(f'Training PHN EPO Loss {num_nodes},{num_items}', epo_loss)
    writer.add_scalar(f'Training PHN NLL {num_nodes},{num_items}', -logprob)
    writer.flush()
