import os.path
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from agent.agent import Agent
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv

def setup(args):
    # similar to Attention learn routing default
    agent = Agent(n_heads=8,
                 num_static_features=3,
                 num_dynamic_features=4,
                 n_gae_layers=3,
                 embed_dim=128,
                 gae_ff_hidden=128,
                 tanh_clip=10,
                 device=args.device)
    critic = Agent(n_heads=8,
                 num_static_features=3,
                 num_dynamic_features=4,
                 n_gae_layers=3,
                 embed_dim=128,
                 gae_ff_hidden=128,
                 tanh_clip=10,
                 device=args.device)        
    agent_opt = torch.optim.AdamW(agent.parameters(), lr=args.lr)
    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    agent_title = args.title + str(args.weight_idx) + "_" + str(args.total_weight)
    last_agent_title = args.title + str(args.weight_idx-1) + "_" + str(args.total_weight)
    model_summary_dir = summary_dir/agent_title
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=model_summary_dir.absolute())

    agent_title = args.title + str(args.weight_idx) + "_" + str(args.total_weight)
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/agent_title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")

    # load previous weight idx agent as initial params, if > 1 
    if args.weight_idx > 1:
        last_checkpoint_dir = pathlib.Path(".")/checkpoint_root/last_agent_title
        last_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        last_checkpoint_path = last_checkpoint_dir/(args.title+".pt")        
        last_checkpoint = torch.load(last_checkpoint_path.absolute(), map_location=torch.device(args.device))
        agent.load_state_dict(last_checkpoint["agent_state_dict"])
        critic.load_state_dict(last_checkpoint["agent_state_dict"])

    checkpoint = None
    if os.path.isfile(checkpoint_path.absolute()):
        checkpoint = torch.load(checkpoint_path.absolute(), map_location=torch.device(args.device))
    else:
        print("CHECKPOINT NOT FOUND! new run?")

    last_epoch = 0
    crit_ws_cost_list = None
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent_opt_state_dict = checkpoint["agent_opt_state_dict"]
        crit_ws_cost_list = checkpoint["crit_ws_cost_list"]
        agent_opt.load_state_dict(agent_opt_state_dict)
        last_epoch = checkpoint["epoch"]

    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    test_batch = next(iter(test_dataloader))
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)

    return agent, agent_opt, critic, last_epoch, writer, test_env, crit_ws_cost_list
