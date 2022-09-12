import torch

from transformer.agent import Agent
from ttp.ttp_env import TTPEnv

def solve(agent: Agent, env: TTPEnv):
    logprobs = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    sum_entropies = torch.zeros((env.batch_size,), device=agent.device, dtype=torch.float32)
    static_features, dynamic_features, eligibility_mask = env.begin()
    static_features = torch.from_numpy(static_features).to(agent.device)
    dynamic_features = torch.from_numpy(dynamic_features).to(agent.device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
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
        previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :].unsqueeze(1)
        selected_idx, logp, entropy = agent(static_embeddings[is_not_finished],
                                   graph_embeddings[is_not_finished],
                                   previous_embeddings,
                                   dynamic_features[is_not_finished],
                                   glimpse_V[:, is_not_finished, :, :],
                                   glimpse_K[:, is_not_finished, :, :],
                                   logits_K[is_not_finished],
                                   eligibility_mask[is_not_finished])
        #save logprobs
        logprobs[is_not_finished] += logp
        sum_entropies[is_not_finished] += entropy
        dynamic_features, eligibility_mask = env.act(active_idx, selected_idx)
        dynamic_features = torch.from_numpy(dynamic_features).to(agent.device)
        eligibility_mask = torch.from_numpy(eligibility_mask).to(agent.device)
        prev_selected_idx[active_idx] = selected_idx

    # get total profits and tour lenghts
    tour_list, item_selection, tour_lengths, total_profits, total_cost = env.finish()
    exit()
    return tour_list, item_selection, tour_lengths, total_profits, total_cost, logprobs, sum_entropies
