import os.path
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from agent.agent import Agent
from policy.r1_nes import R1_NES
from ttp.ttp_dataset import TTPDataset, read_prob, prob_list_to_env
from ttp.ttp_env import TTPEnv

def load_validation_env_list(num_instance_per_config=3):
    num_nodes_list = [20,30]
    nipc_list = [1,3,5]
    num_ic = 3
    config_list = [(nn, nipc, ic) for nn in num_nodes_list for nipc in nipc_list for ic in range(num_ic)]
    data_root = "data_full" 
    validation_dir = pathlib.Path(".")/data_root/"validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    env_list = []
    for config in config_list:
        nn, nipc, ic = config[0],config[1],config[2]
        prob_list = []
        for idx in range(num_instance_per_config):
            dataset_name = "nn_"+str(nn)+"_nipc_"+str(nipc)+"_ic_"+str(ic)+"_"+str(idx)
            dataset_path = validation_dir/(dataset_name+".pt")
            prob = read_prob(dataset_path=dataset_path)
            prob_list += [prob]
        env = prob_list_to_env(prob_list)


def setup_r1_nes(args):
    agent = Agent(n_heads=8,
                 num_static_features=3,
                 num_dynamic_features=4,
                 n_gae_layers=3,
                 embed_dim=128,
                 gae_ff_hidden=128,
                 tanh_clip=10,
                 device=args.device)        
    policy = R1_NES(num_neurons=128, num_dynamic_features=4)

    summary_root = "runs"
    summary_dir = pathlib.Path(".")/summary_root
    model_summary_dir = summary_dir/args.title
    model_summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=model_summary_dir.absolute())

    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/(args.title+".pt")
    agent_checkpoint_path = checkpoint_dir/(args.title+"_agent.pt")

    agent_checkpoint = torch.load(agent_checkpoint_path.absolute(), map_location=torch.device("cpu"))
    agent.load_state_dict(agent_checkpoint["agent_state_dict"])
    checkpoint = None
    if os.path.isfile(checkpoint_path.absolute()):
        checkpoint = torch.load(checkpoint_path.absolute())
    else:
        print("CHECKPOINT NOT FOUND! new run?")


    last_epoch = 0
    if checkpoint is not None:
        policy = checkpoint["policy"]
        last_epoch = checkpoint["epoch"]
    test_dataset = TTPDataset(dataset_name=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    test_batch = next(iter(test_dataloader))
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = test_batch
    test_env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
        
    # load validation test env list
    # so that we don't do it one by one, at least one batch at a time
    validation_env_list = load_validation_env_list(num_instance_per_config=3)

    return agent, policy, last_epoch, writer, checkpoint_path, test_env, test_dataset.prob.sample_solutions
