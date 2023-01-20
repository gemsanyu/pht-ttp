import argparse
from collections import Counter
import pathlib
import sys

import matplotlib.pyplot as plt
import torch

from policy.non_dominated_sorting import fast_non_dominated_sort
from policy.utils import get_crowding_distance

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
                        default=100,
                        help="number of target nondom solutions")
    
    return parser.parse_args(sys.argv[1:])

def sort_by_crowding_distance(score_list):
    fmin, _ = torch.min(score_list, dim=0)
    fmax, _ = torch.max(score_list, dim=0)
    crowding_distance_list = get_crowding_distance(
            score_list, fmax, fmin)
    num_sample = len(score_list)
    idx_list = list(range(num_sample))
    # idx_list.sort(key=cmp_key)
    idx_list.sort(key=lambda i: (-crowding_distance_list[i]))
    idx_list = torch.Tensor(idx_list)
    # rank by argsort one more time
    idx_list = torch.argsort(idx_list)
    return idx_list


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
    _U = unique_solution_list.clone().numpy()
    nondom_idx = fast_non_dominated_sort(_U)[0]
    nondom_solution_list = _U[nondom_idx,:]
    if len(nondom_solution_list)>args.num_target_solutions:
        cd_idx = sort_by_crowding_distance(torch.from_numpy(nondom_solution_list))
        nondom_solution_list = nondom_solution_list[cd_idx, :]
        nondom_solution_list = nondom_solution_list[:args.num_target_solutions,:]
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