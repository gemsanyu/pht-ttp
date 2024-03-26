import argparse
from multiprocessing import Pool
import pathlib
import pickle
import random
import sys
from typing import Optional, List

from ttp.ttp import TTP

ITEM_CORR_DICT = {  0:"uncorrelated",
                    1:"uncorrelated, similar weights",
                    2:"bounded strongly corr"
                }

def generate(num_nodes, num_items_per_city, item_correlation, idx, dataseed=None):
    capacity_factor = random.randint(1,10)

    problem = TTP(dataset_name="eil101")
    problem.num_items_per_city = num_items_per_city
    problem.item_correlation = item_correlation
    problem.generate_items_only()

    data_root = "data_full" 
    data_dir = pathlib.Path(".")/data_root/"test-big-m-101"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "nn_"+str(problem.num_nodes)+"_nipc_"+str(num_items_per_city)+"_ic_"+str(item_correlation)+"_"+str(idx)
    dataset_path = data_dir/(dataset_name+".pt")
    dataset_text_path = data_dir/(dataset_name+".txt")
    with open(dataset_path.absolute(), "wb") as handle:
        pickle.dump(problem, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dataset_text_path.absolute(), "w") as f:
        f.write("PROBLEM NAME:\t"+dataset_name+"\n")
        f.write("KNAPSACK DATA TYPE:\t"+ITEM_CORR_DICT[item_correlation]+"\n")
        f.write("DIMENSION:\t"+str(num_nodes)+"\n")

        f.write("NUMBER OF ITEMS:\t"+str(problem.num_items)+"\n")
        f.write("CAPACITY OF KNAPSACK:\t"+str(int(problem.max_cap))+"\n")
        f.write("MIN SPEED:\t0.1\n")
        f.write("MAX SPEED:\t1\n")
        f.write("RENTING RATION:\t"+str(problem.renting_rate)+"\n")
        f.write("EDGE_WEIGHT_TYPE:\tCEIL_2D\n")
        f.write("NODE_COORD_SECTION	(INDEX, X, Y):\n")
        coords = problem.location_data.coords
        for i in range(num_nodes):
            x = str(int(coords[i,0].item()))
            y = str(int(coords[i,1].item()))
            f.write(str(i+1)+" "+x+" "+y+"\n")
        f.write("ITEMS SECTION	(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):\n")
        for i in range(problem.num_items):
            city_idx = str(int(problem.item_city_idx[i].item()+1))
            profit = str(int(problem.profit_data.profits[i].item()))
            weight = str(int(problem.weight_data.weights[i].item()))
            f.write(str(i+1)+" "+str(profit)+" "+str(weight)+" "+str(city_idx)+"\n")


def run():
    generate_args = [(101, nic, ic, 0, "eil101") for nic in [20,30,50] for ic in range(3)]
    generate(101,20,0,0,"eil101")
    # with Pool(processes=4) as pool:
        # L = pool.starmap(generate, generate_args)

if __name__ == "__main__":
    run()