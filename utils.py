from typing import NamedTuple

import torch
from tqdm import tqdm
from scipy.stats import ttest_rel

from agent.agent import Agent
from agent.critic import Critic
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

def solve(agent: Agent, env: TTPEnv):
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    static_features, dynamic_features, eligibility_mask = env.begin()
    static_features =  static_features.to(agent.device)
    dynamic_features = dynamic_features.to(agent.device)
    eligibility_mask = eligibility_mask.to(agent.device)
    # compute fixed static embeddings and graph embeddings once for reusage
    static_embeddings, graph_embeddings = agent.gae(static_features)
    # similarly, compute glimpse_K, glimpse_V, and logits_K once for reusage
    glimpse_K, glimpse_V, logits_K = agent.project_embeddings(static_embeddings).chunk(3, dim=-1)
    # glimpse_K awalnya batch_size, num_items, embed_dim
    # ubah ke batch_size, num_items, n_heads, key_dim
    # terus permute agar jadi n_heads, batch_size, num_items, key_dim
    glimpse_K = glimpse_K.unsqueeze(2).view(env.batch_size, env.num_items+env.num_nodes, agent.n_heads, agent.embed_dim//agent.n_heads)
    glimpse_K = glimpse_K.permute(2,0,1,3)
    glimpse_V = glimpse_V.unsqueeze(2).view(env.batch_size, env.num_items+env.num_nodes, agent.n_heads, agent.embed_dim//agent.n_heads)
    glimpse_V = glimpse_V.permute(2,0,1,3)
    
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=agent.device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    while torch.any(eligibility_mask):
        is_not_finished = torch.any(eligibility_mask, dim=1)
        active_idx = is_not_finished.nonzero().long().squeeze(1)
        previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :]
        selected_idx, logp, entropy = agent(static_embeddings[is_not_finished],
                                   graph_embeddings[is_not_finished],
                                   previous_embeddings,
                                   dynamic_features[is_not_finished],
                                   glimpse_V[:, is_not_finished, :, :],
                                   glimpse_K[:, is_not_finished, :, :],
                                   logits_K[is_not_finished],
                                   eligibility_mask[is_not_finished])
        # print(logp)
        #save logprobs
        logprobs[is_not_finished] += logp
        sum_entropies[is_not_finished] += entropy
        dynamic_features, eligibility_mask = env.act(active_idx, selected_idx)
        dynamic_features = dynamic_features.to(agent.device)
        eligibility_mask = eligibility_mask.to(agent.device)
        prev_selected_idx[active_idx] = selected_idx

    # get total profits and tour lenghts
    tour_list, item_selection, tour_lengths, total_profits, total_cost = env.finish()
    return tour_list, item_selection, tour_lengths, total_profits, total_cost, logprobs, sum_entropies

        
def train_epoch(agent, agent_opt, checkpoint_path, last_step, current_epoch, writer, train_dataloader: torch.utils.data.DataLoader, eval_batch):
    step = last_step
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="step", position=1):
    # for batch_idx, batch in enumerate(train_dataloader):
        step += 1
        agent_loss, entropy_loss = train_batch(agent, agent_opt, batch)
        train_dataloader.dataset.new_num_items_per_city()
        if step % 10 == 0:
            eval_result = evaluate(agent, eval_batch)
            tour_list, item_selection, tour_length, total_profit, total_cost = eval_result
            write_progress(tour_length, total_profit, total_cost, agent_loss, entropy_loss, step, writer)
            save(agent, agent_opt, current_epoch, step, checkpoint_path)
                
    return step

def train_batch(agent, agent_opt, batch):
    agent.train()
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask = batch
    env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask)
    
    tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, entropy = solve(agent, env)
    # critic costs is total profit if all items are taken
    critic_costs = torch.sum(profits, dim=-1)
    agent_loss, entropy_loss = update(agent, agent_opt, total_costs, critic_costs, logprobs, entropy)
    return agent_loss, entropy_loss


def update(agent, agent_opt, total_costs, critic_costs, logprobs, entropy):
    # critic costs is total profit if all items are taken
    # so smaller advangae is better, 
    # if crit costs < total costs, then adv is negative (small)
    advantage = (critic_costs - total_costs).to(agent.device) # total costs
    # standardize advantage
    advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
    agent_loss = ((advantage.detach())*logprobs).mean()
    entropy_loss = -entropy.mean()
    loss = agent_loss + 0.02*entropy_loss
    
    #update agent
    agent_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1)
    agent_opt.step()

    return agent_loss.item(), entropy_loss.item()

def evaluate(agent, batch):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask = batch
    env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask)
    agent.eval()
    with torch.no_grad():
        tour_list, item_selection, tour_length, total_profit, total_cost, _, _ = solve(agent, env)

    return tour_list, item_selection, tour_length.item(), total_profit.item(), total_cost.item()


def save(agent: Agent, agent_opt:torch.optim.Optimizer,epoch, step, checkpoint_path):
    checkpoint = {
        "agent_state_dict":agent.state_dict(),
        "agent_opt_state_dict":agent_opt.state_dict(),
        "epoch":epoch,
        "step":step
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())


def write_progress(tour_length, total_profit, total_cost, agent_loss, entropy_loss, step, writer):
    # note the parameters
    writer.add_scalar("Tour Length", tour_length, step)
    writer.add_scalar("Total Profit", total_profit, step)
    writer.add_scalar("Total Cost", total_cost, step)
    writer.add_scalar("Agent Loss", agent_loss, step)
    writer.add_scalar("Entropy Loss", entropy_loss, step)
    writer.flush()
