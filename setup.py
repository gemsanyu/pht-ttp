import os.path
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from agent.agent import Agent
from ttp.ttp_dataset import TTPDataset

def setup(args):
    # similar to Attention learn routing default
    agent = Agent(n_heads=8,
                 n_gae_layers=3,
                 input_dim=7,
                 embed_dim=128,
                #  embed_dim=64,
                 gae_ff_hidden=512,
                #  gae_ff_hidden=64,
                 tanh_clip=10,
                 device=args.device)    
    agent_opt = torch.optim.AdamW(agent.parameters(), lr=1e-4)
    
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
        checkpoint = torch.load(checkpoint_path.absolute())
    else:
        print("CHECKPOINT NOT FOUND! new run?")

    last_epoch = 0
    last_step = 0
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent_opt.load_state_dict(checkpoint["agent_opt_state_dict"])
        last_epoch = checkpoint["epoch"]
        last_step = checkpoint["step"]

    train_dataset = TTPDataset(num_samples=1000)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)

    eval_dataset = TTPDataset(dataset_name=args.dataset_name)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)
    eval_batch = next(iter(eval_dataloader))

    return agent, agent_opt, last_epoch, last_step, writer, checkpoint_path, train_dataloader, eval_batch
