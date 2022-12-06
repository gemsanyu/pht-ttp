import argparse
from collections import Counter
import pathlib
import sys

import matplotlib.pyplot as plt
import torch

from policy.utils import get_domination_fronts
from policy.utils import get_nondominated_rank

def get_args():
    parser = argparse.ArgumentParser(description='TTP-MORL')
    # GENERAL
    parser.add_argument('--dataset-name',
                        type=str,
                        default="a280-n279",
                        help="dataset's name for real testing")
    parser.add_argument('--title',
                        type=str,
                        default="att_phn",
                        help="title for experiment")
    parser.add_argument('--num-target-solutions',
                        type=int,
                        default=50,
                        help="number of target nondom solutions")
    
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    # read from results based on dataset name and title
    args = get_args()
    results_dir = pathlib.Path(".")/"results"
    model_result_dir = results_dir/args.title
    model_result_dir.mkdir(parents=True, exist_ok=True)
    nondom_result_dir = model_result_dir/"cleaned"
    y_file_path = model_result_dir/(args.title+"_"+args.dataset_name+".f")
    with open(y_file_path.absolute(), "r") as y_file:
        solution_list = []
        lines = y_file.readlines()
        for i, line in enumerate(lines):
            strings = line.split()
            tour_length = float(strings[0])
            profit = float(strings[1])
            solution_list += [tuple([tour_length, -profit])]
    unique_solution_counter = Counter(solution_list)
    unique_solution_list = []
    for k,v in enumerate(unique_solution_counter):
        unique_solution_list += [[v[0], v[1]]]
    unique_solution_list = torch.tensor(unique_solution_list)
    fronts_idx_list = get_domination_fronts(unique_solution_list)
    nondom_solution_list = None
    for front_idx in fronts_idx_list:
        if nondom_solution_list is None:
            nondom_solution_list = unique_solution_list[front_idx]
        else:
            nondom_solution_list = torch.cat([nondom_solution_list, unique_solution_list[front_idx]])
        if len(nondom_solution_list) >= args.num_target_solutions:
            break
    nondom_rank = get_nondominated_rank(nondom_solution_list)
    nondom_solution_list = nondom_solution_list[nondom_rank][:args.num_target_solutions]
    nondom_result_dir.mkdir(parents=True, exist_ok=True)
    nondom_y_file_path = nondom_result_dir/(args.title+"_"+args.dataset_name+".f")
    with open(nondom_y_file_path.absolute(), "w") as nondom_y_file:
        for i in range(len(nondom_solution_list)):
            tour_length = "{:.16f}".format(nondom_solution_list[i,0].item())
            total_profit = "{:.16f}".format(-nondom_solution_list[i,1].item())
            nondom_y_file.write(tour_length+" "+total_profit+"\n")

    # plt.scatter(unique_solution_list[:, 0], unique_solution_list[:, 1], c='b')
    # plt.scatter(nondom_solution_list[:, 0], nondom_solution_list[:, 1], c='r', marker="v")
    # plt.show()