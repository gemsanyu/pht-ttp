import os
import pathlib
import random
import sys


import numpy as np
import torch
from tqdm import tqdm

from arguments import get_parser
from setup import setup_r1_nes
from utils import solve_decode_only

CPU_DEVICE = torch.device("cpu")
MASTER = 0
EVALUATOR = 1

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    return args

@torch.no_grad()
def test_one_epoch(agent, policy, test_env, x_file, y_file, pop_size=200):
    agent.eval()
    # reuse the static embeddings
    #get static embeddings first, it can be widely reused
    static_features, _, _, _ = test_env.begin()
    static_features = torch.from_numpy(static_features).to(CPU_DEVICE)
    static_embeddings, graph_embeddings = agent.gae(static_features)
    static_embeddings = static_embeddings.to(agent.device)
    graph_embeddings = graph_embeddings.to(agent.device)

    param_dict_list, sample_list = policy.generate_random_parameters(n_sample=pop_size, use_antithetic=False)
    
    for param_dict in tqdm(param_dict_list):
        tour_list, item_selection, tour_lengths, total_profits, total_costs, logprobs, sum_entropies = solve_decode_only(agent, test_env, static_embeddings, graph_embeddings, param_dict)
        node_order_str = ""
        for i in tour_list[0]:
            node_order_str+= str(i.item()) + " "
        x_file.write(node_order_str+"\n")
        item_selection_str = ""
        for i in item_selection[0]:
            item_selection_str += (str(int(i.item()))) + " "
        x_file.write(item_selection_str+"\n")

        tour_length = "{:.16f}".format(tour_lengths[0].item())
        total_profit = "{:.16f}".format(total_profits[0].item())
        y_file.write(tour_length+" "+total_profit+"\n")
        print(tour_length+" "+total_profit+"\n")

def test(args):
    agent, policy, last_epoch, writer, checkpoint_path, test_env, sample_solutions = setup_r1_nes(args)
    agent.gae = agent.gae.cpu()
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file:
        test_one_epoch(agent, policy, test_env, x_file, y_file)


if __name__=='__main__':
    args = prepare_args()
    #torch.set_num_threads(os.cpu_count()-4)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    test(args)
