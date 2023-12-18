import argparse
from multiprocessing import Pool
import pathlib
import pickle
import sys
from typing import List

from ttp.ttp import TTP

def get_args():
    parser = argparse.ArgumentParser(description='generate dataset')
    parser.add_argument('--dataseed',
                        type=str,
                        nargs="?",
                        default="eil76_n225_uncorr_05",
                        help="dataset's name for real testing")

    parser.add_argument('--num-dataset',
                        type=int,
                        default=38,
                        help="num of datasets generated per config")

    parser.add_argument('--num-nodes-list',
                        type=int,
                        nargs="+",
                        default=[10,20,30],
                        help="num of nodes in dataset")

    parser.add_argument('--num-items-per-city',
                        type=int,
                        nargs="+",
                        default=[1,3,5],
                        help="num items per city") 

    parser.add_argument('--item-correlation-list',
                        type=int,
                        nargs="+",
                        default=list(range(3)),
                        help="item correlations") 

    parser.add_argument('--capacity-factor-list',
                        type=int,
                        nargs="+",
                        default=[i for i in range(1,11)],
                        help="capacity factor later be divided by 11 in TTP")   
     
    parser.add_argument('--mode',
                        type=str,
                        default="training",
                        help="where to use the generated instance, training/validation")   
    args = parser.parse_args(sys.argv[1:])
    return args


def generate(num_nodes, num_items_per_city, item_correlation, capacity_factor, idx, dataseed=None, mode="validation"):

    problem = TTP(num_nodes=num_nodes, num_items_per_city=num_items_per_city, item_correlation=item_correlation, capacity_factor=capacity_factor, dataseed=dataseed)
    data_root = "data_full" 
    data_dir = pathlib.Path(".")/data_root/mode
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "nn_"+str(num_nodes)+"_nipc_"+str(num_items_per_city)+"_ic_"+str(item_correlation)+"_cf_"+str(capacity_factor)+"_"+str(idx)
    dataset_path = data_dir/(dataset_name+".pt")

    with open(dataset_path.absolute(), "wb") as handle:
        pickle.dump(problem, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run(args):
    generate_args = [(nn, nic, ic, cf, idx, args.dataseed, args.mode) for nn in args.num_nodes_list for nic in args.num_items_per_city for ic in args.item_correlation_list for cf in args.capacity_factor_list for idx in range(args.num_dataset)]
    with Pool(processes=20) as pool:
        L = pool.starmap(generate, generate_args)

if __name__ == "__main__":
    args = get_args()
    run(args)

    
