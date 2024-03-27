import os.path
import pathlib
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from agent.agent import Agent
from agent.encoder import Encoder
from policy.r1_nes import R1_NES
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv


def setup_r1_nes(args, load_best=False):
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
    policy = R1_NES(num_neurons=args.encoder_size,
                    ld=args.ld,
                    negative_hv=args.negative_hv,
                    pop_size=args.pop_size,
                    lr=args.lr)

    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    model_summary_dir = summary_dir/args.title
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=model_summary_dir.absolute())

    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")
    if load_best:
        checkpoint_path = checkpoint_dir/(args.title+"_best.pt")
    agent_checkpoint_path = checkpoint_dir/(args.title+"_agent.pt")

    agent_checkpoint = torch.load(agent_checkpoint_path.absolute(), map_location=args.device)
    agent.load_state_dict(agent_checkpoint["agent_state_dict"])
    checkpoint = None
    if os.path.isfile(checkpoint_path.absolute()):
        checkpoint = torch.load(checkpoint_path.absolute(), map_location=torch.device("cpu"))
    else:
        print("CHECKPOINT NOT FOUND! new run?")
    policy.copy_to_mu(agent)

    last_epoch = 0
    training_nondom_list = None
    validation_nondom_list = None
    best_f_list = None
    if checkpoint is not None:
        policy = checkpoint["policy"]
        last_epoch = checkpoint["epoch"]
        training_nondom_list = checkpoint["training_nondom_list"]
        validation_nondom_list = checkpoint["validation_nondom_list"]
        best_f_list = checkpoint["best_f_list"]

    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    _, test_batch = next(iter(test_dataloader))
    # coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    # test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
        
    # trace the model because it does not need any grad
    agent.eval()
    agent, encoder = trace_model(agent, policy)
    return agent, encoder, policy, training_nondom_list, validation_nondom_list, best_f_list, last_epoch, writer, checkpoint_path, test_batch, test_dataset.prob.sample_solutions

@torch.no_grad()
def trace_model(agent:Agent, policy:R1_NES)-> Tuple[Agent, Encoder]:
    device = agent.device
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=policy.pop_size, use_antithetic=False)
    param_dict = param_dict_list[0]
    param_dict["v1"] = param_dict["v1"].to(device)

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