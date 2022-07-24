import random
import sys

import numpy as np
import torch

from arguments import get_parser
from setup import setup
from utils import train_epoch

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

def run(args):
    agent, agent_opt, last_epoch, last_step, writer, checkpoint_path, train_dataloader, eval_batch = setup(args)
    for epoch in range(last_epoch, args.max_epoch):
        last_step = train_epoch(agent, agent_opt, checkpoint_path, last_step, epoch, writer, train_dataloader, eval_batch)

if __name__=='__main__':
    args = prepare_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)