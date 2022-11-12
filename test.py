import pathlib
import random
import sys

import numpy as np
import torch
from tqdm import tqdm


from arguments import get_parser
from setup import setup_phn
from utils import solve

CPU_DEVICE = torch.device("cpu")

def prepare_args():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device(args.device)
    return args

@torch.no_grad()
def test(agent, phn, test_env, x_file, y_file, n_solutions=200):
    agent.eval()
    phn.eval()
    ray_list = [torch.tensor([[float(i)/n_solutions,1-float(i)/n_solutions]]) for i in range(n_solutions)]
    for ray in tqdm(ray_list, desc="Testing"):
        param_dict = phn(ray.to(agent.device))
        tour_list, item_selection, tour_length, total_profit, total_cost, logprob, sum_entropies = solve(agent, test_env, param_dict)
        node_order_str = ""
        for i in tour_list[0]:
            node_order_str+= str(i.item()) + " "
        x_file.write(node_order_str+"\n")
        item_selection_str = ""
        for i in item_selection[0]:
            item_selection_str += (str(int(i.item()))) + " "
        x_file.write(item_selection_str+"\n")

        tour_length = "{:.16f}".format(tour_length[0].item())
        total_profit = "{:.16f}".format(total_profit[0].item())
        y_file.write(tour_length+" "+total_profit+"\n")
        print(tour_length+" "+total_profit+"\n")    

def run(args):
    agent, phn, phn_opt, last_epoch, writer, checkpoint_path, test_env, test_sample_solutions = setup_phn(args)
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    x_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".x")
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    
    with open(x_file_path.absolute(), "a+") as x_file, open(y_file_path.absolute(), "a+") as y_file:
        test(agent, phn, test_env, x_file, y_file)

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    args = prepare_args()
    # torch.set_num_threads(os.cpu_count())
    torch.set_num_threads(16)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run(args)