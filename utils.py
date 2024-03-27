import os
import pathlib
import sys
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from arguments import get_parser
from agent.agent import Agent, select
from agent.encoder import Encoder
from policy.hv import Hypervolume
from policy.normalization import normalize
from policy.non_dominated_sorting import fast_non_dominated_sort
from ttp.ttp_env import TTPEnv

CPU_DEVICE = torch.device('cpu')
def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

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

def encode(encoder:Encoder, static_features, num_nodes, num_items, batch_size):
    static_features =  torch.from_numpy(static_features).to(encoder.device)
    item_static_embeddings = encoder.item_static_encoder(static_features[:,:num_items,:])
    depot_static_embeddings = encoder.depot_init_embed.expand((batch_size,1,-1))
    # node_static_embeddings = agent.node_init_embed.expand((batch_size, num_nodes-1, -1))
    node_static_embeddings = encoder.node_encoder(static_features[:,num_items+1:,:])
    static_embeddings = torch.cat([item_static_embeddings, depot_static_embeddings, node_static_embeddings], dim=1)
    return static_embeddings

def solve_decode_only(agent: Agent, encoder: Encoder, env: TTPEnv, static_embeddings, param_dict=None, normalized=False):
    if param_dict is not None:
        param_dict["v1"] = param_dict["v1"].to(encoder.device)
    logprobs = torch.zeros((env.batch_size,), device=encoder.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=encoder.device, dtype=torch.float32)
    last_pointer_hidden_states = torch.zeros((2, env.batch_size, 128), device=encoder.device, dtype=torch.float32)
    static_features, dynamic_features, eligibility_mask = env.begin()
    num_items = env.num_items
    dynamic_features = torch.from_numpy(dynamic_features).to(encoder.device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(encoder.device)
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=encoder.device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    # initially pakai initial input
    previous_embeddings = encoder.inital_input.repeat_interleave(env.batch_size, dim=0)
    first_turn = True
    while torch.any(eligibility_mask):
        is_not_finished = torch.any(eligibility_mask, dim=1)
        active_idx = is_not_finished.nonzero().long().squeeze(1)
        if not first_turn:
            previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :]
            previous_embeddings = previous_embeddings.unsqueeze(1)
        next_pointer_hidden_states = last_pointer_hidden_states
        item_dynamic_embeddings = encoder.item_dynamic_encoder(dynamic_features[:,:num_items,:])
        node_dynamic_embeddings = encoder.node_dynamic_encoder(dynamic_features[:,num_items:,:])
        dynamic_embeddings = torch.cat([item_dynamic_embeddings, node_dynamic_embeddings], dim=1)
        forward_results = agent(last_pointer_hidden_states[:, active_idx, :], static_embeddings[active_idx], dynamic_embeddings[active_idx],eligibility_mask[active_idx], previous_embeddings, param_dict)
        next_pointer_hidden_states[:, active_idx, :], logits, probs = forward_results
        last_pointer_hidden_states = next_pointer_hidden_states
        selected_idx, logprob, entropy = select(probs)
        #save logprobs
        logprobs[active_idx] += logprob
        sum_entropies[active_idx] += entropy
        dynamic_features, eligibility_mask = env.act(active_idx, selected_idx)
        dynamic_features = torch.from_numpy(dynamic_features).to(encoder.device)
        eligibility_mask = torch.from_numpy(eligibility_mask).to(encoder.device)
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


def save_nes(policy, training_nondom_list, validation_nondom_list, best_f_list, epoch, title, best=False):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt")
    if best:
        checkpoint_path = checkpoint_dir/(title+"_best.pt")
        

    checkpoint = {
        "policy":policy,  
        "epoch":epoch,
        "training_nondom_list": training_nondom_list,
        "validation_nondom_list":validation_nondom_list,
        "best_f_list":best_f_list
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

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



def write_training_phn_progress(mean_total_profit, mean_tour_length, profit_loss, tour_length_loss, epo_loss, logprob, num_nodes, num_items, writer):
    writer.add_scalar(f'Training PHN Mean Total Profit {num_nodes},{num_items}', mean_total_profit)
    writer.add_scalar(f'Training PHN Mean Tour Length {num_nodes},{num_items}', mean_tour_length)
    writer.add_scalar(f'Training PHN Profit Loss {num_nodes},{num_items}', profit_loss)
    writer.add_scalar(f'Training PHN Tour Length Loss {num_nodes},{num_items}', tour_length_loss)
    writer.add_scalar(f'Training PHN EPO Loss {num_nodes},{num_items}', epo_loss)
    writer.add_scalar(f'Training PHN NLL {num_nodes},{num_items}', -logprob)
    writer.flush()
