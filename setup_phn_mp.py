import pathlib
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from agent.encoder import Encoder, load_encoder, save_encoder
from agent.agent import Agent
from agent.phn import PHN
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv

def setup_phn_mp(args, load_best=False):   
    phn = PHN(ray_hidden_size=args.ray_hidden_size, 
            num_neurons=args.encoder_size, 
            device=args.device)
    
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")
    if load_best:
        checkpoint_path = checkpoint_dir/(args.title+"_best.pt")
    agent_checkpoint_path = checkpoint_dir/(args.title+"_agent.pt")
    checkpoint = torch.load(checkpoint_path.absolute(), map_location=args.device)
    phn.load_state_dict(checkpoint["phn_state_dict"])
    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    _, test_batch = next(iter(test_dataloader))
    trace_and_save_model(args, phn)
    encoder = load_encoder("cuda")
    return encoder, phn, test_batch

def trace_and_save_model(args, phn):
    agent = Agent(device=args.device,
                  num_static_features=3,
                  num_dynamic_features=4,
                  static_encoder_size=args.encoder_size,
                  dynamic_encoder_size=args.encoder_size,
                  decoder_encoder_size=args.encoder_size,
                  pointer_num_layers=args.pointer_layers,
                  pointer_num_neurons=args.encoder_size,
                  dropout=args.dropout,
                  n_glimpses=args.n_glimpses)   

    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_checkpoint_path = checkpoint_dir/(args.title+"_agent.pt")

    agent_checkpoint = torch.load(agent_checkpoint_path.absolute(), map_location=args.device)
    agent.load_state_dict(agent_checkpoint["agent_state_dict"])
    # trace the model because it does not need any grad
    agent.eval()
    agent, encoder = trace_model(agent, phn)
    torch.jit.save(agent, "traced_agent-nrw.pt")
    save_encoder(encoder)

@torch.no_grad()
def trace_model(agent:Agent, phn)-> Tuple[Agent, Encoder]:
    device = agent.device
    ray_list = [torch.tensor([[float(i)/2,1-float(i)/2]]) for i in range(2)]
    param_dict = phn(ray_list[0].to(phn.device))

    small_test_dataset = TTPDataset(dataset_name="eil51_n50")
    small_test_dataloader = DataLoader(small_test_dataset, batch_size=1)
    _, small_test_batch = next(iter(small_test_dataloader))
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
    next_pointer_hidden_states = last_pointer_hidden_states
    item_dynamic_encoder = torch.jit.trace(agent.item_dynamic_encoder, (dynamic_features[:,:num_items,:]))
    node_dynamic_encoder = torch.jit.trace(agent.node_dynamic_encoder, (dynamic_features[:,num_items:,:]))
    item_dynamic_embeddings = item_dynamic_encoder(dynamic_features[:,:num_items,:])
    node_dynamic_embeddings = node_dynamic_encoder(dynamic_features[:,num_items:,:])
    dynamic_embeddings = torch.cat([item_dynamic_embeddings, node_dynamic_embeddings], dim=1)
    agent = torch.jit.trace(agent, (last_pointer_hidden_states[:, active_idx, :], static_embeddings[active_idx], dynamic_embeddings[active_idx],eligibility_mask[active_idx], previous_embeddings, param_dict))
    
    encoder = Encoder(
        device,
        item_static_encoder,
        node_encoder,
        item_dynamic_encoder,
        node_dynamic_encoder,
        depot_init_embed,
        initial_input)
    return agent, encoder