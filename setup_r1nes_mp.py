import os.path
import pathlib
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from agent.agent import Agent
from agent.encoder import Encoder, save_encoder, load_encoder
from policy.r1_nes import R1_NES
from ttp.ttp_dataset import TTPDataset
from ttp.ttp_env import TTPEnv
from setup_r1nes import trace_model



def setup_r1nes_mp(args, load_best=False):
    policy = R1_NES(num_neurons=args.encoder_size,
                    ld=args.ld,
                    negative_hv=args.negative_hv,
                    pop_size=args.pop_size,
                    lr=args.lr)

    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")
    if load_best:
        checkpoint_path = checkpoint_dir/(args.title+"_best.pt")
    checkpoint = torch.load(checkpoint_path.absolute(), map_location=torch.device("cpu"))    
    policy = checkpoint["policy"]

    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    _, test_batch = next(iter(test_dataloader))
    # if not os.path.isfile("traced_agent-nrw.pt"):
    trace_and_save_model(args, policy)
    encoder = load_encoder("cuda")
    return encoder, policy, test_batch


def trace_and_save_model(args, policy:R1_NES):
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
    agent, encoder = trace_model(agent, policy)
    torch.jit.save(agent, "traced_agent-nrw.pt")
    save_encoder(encoder)