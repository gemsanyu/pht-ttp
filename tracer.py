import os.path
import pathlib
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from agent.agent import Agent
from agent.encoder import Encoder, save_encoder
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv


@torch.no_grad()
def trace_model(agent: Agent)->Tuple[Agent, Encoder]:
    agent.eval()
    device = agent.device
    small_test_dataset = TTPDataset(dataset_name="eil51_n50")
    small_test_dataloader = DataLoader(small_test_dataset, batch_size=1)
    small_test_batch = next(iter(small_test_dataloader))
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = small_test_batch
    env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    num_items = env.num_items
    
    last_pointer_hidden_states = torch.zeros((2, env.batch_size, 128), device=agent.device, dtype=torch.float32)
    static_features, dynamic_features, eligibility_mask = env.begin()
    static_features =  torch.from_numpy(static_features).to(device)
    item_static_encoder = torch.jit.trace(agent.item_static_encoder, (static_features[:,:num_items,:]))
    item_static_embeddings = item_static_encoder(static_features[:,:num_items,:])
    depot_init_embed = agent.depot_init_embed
    depot_static_embeddings = depot_init_embed.expand((env.batch_size,1,-1))
    node_encoder = torch.jit.trace(agent.node_encoder,(static_features[:,num_items+1:,:]))
    node_static_embeddings = node_encoder(static_features[:,num_items+1:,:])
    static_embeddings = torch.cat([item_static_embeddings, depot_static_embeddings, node_static_embeddings], dim=1)
    
    dynamic_features = torch.from_numpy(dynamic_features).to(device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(device)
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    # initially pakai initial input
    initial_input = agent.inital_input
    previous_embeddings = initial_input.repeat_interleave(env.batch_size, dim=0)
    first_turn = True
    is_not_finished = torch.any(eligibility_mask, dim=1)
    active_idx = is_not_finished.nonzero().long().squeeze(1)
    if not first_turn:
        previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :]
        previous_embeddings = previous_embeddings.unsqueeze(1)
    item_dynamic_encoder = torch.jit.trace(agent.item_dynamic_encoder, (dynamic_features[:,:num_items,:]))
    node_dynamic_encoder = torch.jit.trace(agent.node_dynamic_encoder, (dynamic_features[:,num_items:,:]))
    item_dynamic_embeddings = item_dynamic_encoder(dynamic_features[:,:num_items,:])
    node_dynamic_embeddings = node_dynamic_encoder(dynamic_features[:,num_items:,:])
    dynamic_embeddings = torch.cat([item_dynamic_embeddings, node_dynamic_embeddings], dim=1)
    agent = torch.jit.trace(agent, (last_pointer_hidden_states[:, active_idx, :], static_embeddings[active_idx], dynamic_embeddings[active_idx],eligibility_mask[active_idx], previous_embeddings))
    
    encoder = Encoder(
        device,
        item_static_encoder,
        node_encoder,
        item_dynamic_encoder,
        node_dynamic_encoder,
        depot_init_embed,
        initial_input)
    return agent, encoder

def load_agent(title, weight_idx, total_weight, device=torch.device("cuda"), is_best=True):
    agent = Agent(device=device,
                  num_static_features=3,
                  num_dynamic_features=4,
                  static_encoder_size=128,
                  dynamic_encoder_size=128,
                  decoder_encoder_size=128,
                  pointer_num_layers=2,
                  pointer_num_neurons=128,
                  dropout=0.2,
                  n_glimpses=1)     
    agent_title = title + str(weight_idx) + "_" + str(total_weight)
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/agent_title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(title+".pt_best")
    try:
        checkpoint = torch.load(checkpoint_path.absolute(), map_location=device)
    except FileNotFoundError:
        checkpoint_path = checkpoint_dir/(title+".pt")
        checkpoint = torch.load(checkpoint_path.absolute(), map_location=device)    
    agent.load_state_dict(checkpoint["agent_state_dict"])
    return agent

def trace_and_save_model(title, weight_idx, total_weight):
    agent = load_agent(title, weight_idx, total_weight)
    agent, encoder = trace_model(agent)    
    traced_name = title+"_"+str(weight_idx)+"_"+str(total_weight)+"traced_agent.pt"
    traced_dir = pathlib.Path()/"traced_agent"
    traced_dir.mkdir(parents=True, exist_ok=True)
    traced_path = (traced_dir/traced_name).absolute()
    torch.jit.save(agent, traced_path)
    save_encoder(encoder, title, weight_idx, total_weight)