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


def encode(agent, static_features, num_nodes, num_items, batch_size):
    static_features =  torch.from_numpy(static_features).to(agent.device)
    item_static_embeddings = agent.item_static_encoder(static_features[:,:num_items,:])
    depot_static_embeddings = agent.depot_init_embed.expand((batch_size,1,-1))
    # node_static_embeddings = agent.node_init_embed.expand((batch_size, num_nodes-1, -1))
    node_static_embeddings = agent.node_encoder(static_features[:,num_items+1:,:])
    static_embeddings = torch.cat([item_static_embeddings, depot_static_embeddings, node_static_embeddings], dim=1)
    return static_embeddings

def solve_decode_only(agent: Agent, env: TTPEnv, static_embeddings, param_dict=None, normalized=False):
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    last_pointer_hidden_states = torch.zeros((agent.pointer.num_layers, env.batch_size, agent.pointer.num_neurons), device=agent.device, dtype=torch.float32)
    static_features, dynamic_features, eligibility_mask = env.begin()
    dynamic_features = torch.from_numpy(dynamic_features).to(agent.device)
    num_items = env.num_items
    eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=agent.device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    # initially pakai initial input
    previous_embeddings = agent.inital_input.repeat_interleave(env.batch_size, dim=0)
    first_turn = True
    # active_param_dict = param_dict
    while torch.any(eligibility_mask):
        is_not_finished = torch.any(eligibility_mask, dim=1)
        active_idx = is_not_finished.nonzero().long().squeeze(1)
        if not first_turn:
            previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :]
            previous_embeddings = previous_embeddings.unsqueeze(1)
        next_pointer_hidden_states = last_pointer_hidden_states
        item_dynamic_embeddings = agent.item_dynamic_encoder(dynamic_features[:,:num_items,:])
        node_dynamic_embeddings = agent.node_dynamic_encoder(dynamic_features[:,num_items:,:])
        dynamic_embeddings = torch.concatenate([item_dynamic_embeddings, node_dynamic_embeddings], dim=1)
        # dynamic_embeddings = agent.dynamic_encoder(dynamic_features)
        # if param_dict is not None:
        #     active_param_dict = {"v1":param_dict["v1"][active_idx,:,:]}
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

def solve(agent: Agent, env: TTPEnv, param_dict=None, normalized=False):
    static_features, dynamic_features, eligibility_mask = env.begin()
    num_nodes, num_items, batch_size = env.num_nodes, env.num_items, env.batch_size
    static_embeddings = encode(agent, static_features, num_nodes, num_items, batch_size)
    solve_output = solve_decode_only(agent, env, static_embeddings, param_dict)
    return solve_output

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
    adv = critic_total_costs-total_costs # want to maximize
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
    

# def evaluate(agent, batch):
#     coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask = batch
#     env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask)
#     agent.eval()
#     with torch.no_grad():
#         tour_list, item_selection, tour_length, total_profit, total_cost, _, _ = solve(agent, env)

#     return tour_list, item_selection, tour_length.item(), total_profit.item(), total_cost.item()


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