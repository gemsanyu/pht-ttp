import argparse
from multiprocessing import Pool
import pathlib
import pickle
import random
import sys
from typing import Optional, List

from ttp.ttp import TTP

def get_args():
    parser = argparse.ArgumentParser(description='generate dataset')
    parser.add_argument('--dataseed',
                        type=str,
                        nargs="?",
                        default="eil76-n75",
                        help="dataset's name for real testing")

    parser.add_argument('--num-dataset',
                        type=int,
                        default=5,
                        help="num of datasets generated per config")

    parser.add_argument('--num-nodes',
                        type=int,
                        default=30,
                        help="num of nodes in dataset")

    parser.add_argument('--num-items-per-city',
                        type=int,
                        nargs="+",
                        default=[1,3,5],
                        help="num of nodes in dataset")    

    args = parser.parse_args(sys.argv[1:])
    return args


def generate(num_nodes, num_items_per_city, item_correlation, idx, dataseed=None):
    capacity_factor = random.randint(1,10)

    problem = TTP(num_nodes=num_nodes, num_items_per_city=num_items_per_city, item_correlation=item_correlation, capacity_factor=capacity_factor, dataseed=dataseed)
    data_root = "data_full" 
    data_dir = pathlib.Path(".")/data_root/"training"/"sop"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "nn_"+str(num_nodes)+"_nipc_"+str(num_items_per_city)+"_ic_"+str(item_correlation)+"_"+str(idx)
    dataset_path = data_dir/(dataset_name+".pt")

    with open(dataset_path.absolute(), "wb") as handle:
        pickle.dump(problem, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run(args):
    generate_args = [(args.num_nodes, nic, ic, idx, args.dataseed) for nic in args.num_items_per_city for ic in range(3) for idx in range(args.num_dataset)]
    with Pool(processes=2) as pool:
        L = pool.starmap(generate, generate_args)

if __name__ == "__main__":
    args = get_args()
    run(args)

    
