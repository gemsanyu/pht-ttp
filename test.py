import random
import sys

import numpy as np
import torch

from setup import setup_test
from arguments import get_parser
from utils.ttp import TTP

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.policy_device = torch.device(args.policy_device)
    args.actor_device = torch.device(args.actor_device)
    return args

def test(args):
    agent_template, policy, test_problem, x_file, y_file = setup_test(args)
    # if this is the first run, then evaluate first
    evaluate(agent_template, policy, args.test_pop_size, test_problem, x_file, y_file)


def solve(agent_template, param_dict, problem):
    agent_template.load_state_dict(param_dict)
    with torch.no_grad():
        node_order, item_selection = agent_template.forward(problem)
    travel_time = problem.get_total_time(node_order, item_selection)
    total_profit = problem.get_total_profit(item_selection)
    return node_order, item_selection, travel_time, total_profit

def evaluate(agent_template, policy, pop_size, problem, x_file, y_file):
    # evaluating every K iteration
    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    
    for n in range(pop_size):
        # print("EVALUATE",n)
        node_order, item_selection, travel_time, total_profit =\
            solve(agent_template, param_dict_list[n], problem)
        node_order_str = ""
        for i in node_order:
            node_order_str+= str(i.item()) + " "
        x_file.write(node_order_str+"\n")
        item_selection_str = ""
        for i in item_selection:
            item_selection_str += (str(int(i.item()))) + " "
        x_file.write(item_selection_str+"\n")

        travel_time = "{:.16f}".format(travel_time.item())
        total_profit = "{:.16f}".format(total_profit.item())
        y_file.write(travel_time+" "+total_profit+"\n")


def save(policy, epoch, checkpoint_path):
    checkpoint = {
        "policy":policy,
        "epoch":epoch,
    }
    # save twice to prevent failed saving,,, damn
    torch.save(checkpoint, checkpoint_path.absolute())
    checkpoint_backup_path = checkpoint_path.parent /(checkpoint_path.name + "_")
    torch.save(checkpoint, checkpoint_backup_path.absolute())

if __name__=='__main__':
    args = prepare_args()
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    test(args)
