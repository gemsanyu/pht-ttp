import os.path
import pathlib
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from agent.agent import Agent, make_heads
from agent.encoder import Encoder, load_encoder, save_encoder
from policy.r1_nes import R1_NES
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv

def setup_master(args, load_best=False):
    policy = R1_NES(num_neurons=256, 
                    num_dynamic_features=4,
                    ld=args.ld,
                    lr=args.lr,
                    negative_hv=args.negative_hv,
                    pop_size=args.pop_size)

    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")
    if load_best:
        checkpoint_path = checkpoint_dir/(args.title+"_best.pt")
    checkpoint = torch.load(checkpoint_path.absolute())
    policy = checkpoint["policy"]
    trace_and_save_agent(args, policy)
    encoder = load_encoder("cuda")
    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    _, test_batch = next(iter(test_dataloader))

    return encoder, policy, test_batch

def trace_and_save_agent(args, policy):
    agent = Agent(n_heads=8,
                 num_static_features=3,
                 num_dynamic_features=4,
                 n_gae_layers=6,
                 embed_dim=256,
                 gae_ff_hidden=128,
                 tanh_clip=10,
                 device=args.device)      
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_checkpoint_path = checkpoint_dir/(args.title+"_agent.pt")

    agent_checkpoint = torch.load(agent_checkpoint_path.absolute(), map_location=args.device)
    agent.load_state_dict(agent_checkpoint["agent_state_dict"])
    agent, encoder = trace_model(agent, policy)
    torch.jit.save(agent, "traced_agent")
    save_encoder(encoder)


def setup_r1_nes(args, load_best=False):
    encoder, policy, test_batch = setup_master(args, load_best)
    return encoder, policy, test_batch

@torch.no_grad()
def trace_model(agent: Agent, policy: R1_NES)->Tuple[Agent, Encoder]:
    agent.eval()
    device = agent.device
    dataset = TTPDataset(dataset_name="eil51_n50")
    dataloader = DataLoader(dataset, batch_size=1)
    _, batch = next(iter(dataloader))

    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch
    env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    static_features = env.get_static_features()    
    num_nodes, num_items, batch_size = env.num_nodes, env.num_items, env.batch_size
    static_features = torch.from_numpy(static_features).to(device)
    item_init_embedder = torch.jit.trace(agent.item_init_embedder, (static_features[:, :num_items, :]))
    item_init_embed = item_init_embedder(static_features[:, :num_items, :])
    depot_init_embed_ = agent.depot_init_embed
    depot_init_embed = depot_init_embed_.expand(size=(batch_size,1,-1))
    node_init_embedder = torch.jit.trace(agent.node_init_embed, (static_features[:,num_items+1:,:]))
    node_init_embed = node_init_embedder(static_features[:,num_items+1:,:])
    init_embed = torch.cat([item_init_embed, depot_init_embed, node_init_embed], dim=1)
    gae = torch.jit.trace(agent.gae,(init_embed))
    static_embeddings, graph_embeddings = gae(init_embed)
    project_fixed_context = torch.jit.trace(agent.project_fixed_context,(graph_embeddings))
    fixed_context = project_fixed_context(graph_embeddings)
    project_embeddings = torch.jit.trace(agent.project_embeddings, (static_embeddings))
    glimpse_K_static, glimpse_V_static, logits_K_static = project_embeddings(static_embeddings).chunk(3, dim=-1)
    glimpse_K_static = make_heads(glimpse_K_static)
    glimpse_V_static = make_heads(glimpse_V_static)
    
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=4, use_antithetic=False)
    param_dict = param_dict_list[0]
    param_dict["po_weight"] = param_dict["po_weight"].to(device)
    static_features, node_dynamic_features, global_dynamic_features, eligibility_mask = env.begin()
    static_features = torch.from_numpy(static_features).to(device)
    node_dynamic_features = torch.from_numpy(node_dynamic_features).to(device)
    global_dynamic_features = torch.from_numpy(global_dynamic_features).to(device)
    eligibility_mask = torch.from_numpy(eligibility_mask).to(device)
    
    prev_selected_idx = torch.zeros((env.batch_size,), dtype=torch.long, device=device)
    prev_selected_idx = prev_selected_idx + env.num_nodes
    is_not_finished = torch.any(eligibility_mask, dim=1)
    active_idx = is_not_finished.nonzero().long().squeeze(1)
    previous_embeddings = static_embeddings[active_idx, prev_selected_idx[active_idx], :].unsqueeze(1)
    num_items = torch.tensor(env.num_items).to(device)
    agent = torch.jit.trace(agent,(num_items,
                                static_embeddings[is_not_finished],
                                fixed_context[is_not_finished],
                                previous_embeddings,
                                node_dynamic_features[is_not_finished],
                                global_dynamic_features[is_not_finished],    
                                glimpse_V_static[:, is_not_finished, :, :],
                                glimpse_K_static[:, is_not_finished, :, :],
                                logits_K_static[is_not_finished],
                                eligibility_mask[is_not_finished],
                                param_dict))
    encoder = Encoder(
        device,
        gae,
        item_init_embedder,
        depot_init_embed_,
        node_init_embedder,
        project_fixed_context,
        project_embeddings)
    return agent, encoder