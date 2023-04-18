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
    critic =  Agent(n_heads=8,
                 num_static_features=3,
                 num_dynamic_features=4,
                 n_gae_layers=3,
                 embed_dim=128,
                 gae_ff_hidden=128,
                 tanh_clip=10,
                 device=args.device)   
    agent_opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    model_summary_dir = summary_dir/args.title
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=model_summary_dir.absolute())

    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")

    checkpoint = None
    if os.path.isfile(checkpoint_path.absolute()):
        checkpoint = torch.load(checkpoint_path.absolute(), map_location=args.device)
    else:
        print("CHECKPOINT NOT FOUND! new run?")

    last_epoch = 0
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        agent_opt.load_state_dict(checkpoint["agent_opt_state_dict"])
        last_epoch = checkpoint["epoch"]

    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    test_batch = next(iter(test_dataloader))
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp)
        
    return agent, agent_opt, critic, last_epoch, writer, checkpoint_path, test_env
