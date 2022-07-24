from multiprocessing import Pool
import pathlib
import pickle
import random


from ttp.ttp import TTP

def generate(num_nodes, num_items_per_city, prob_idx):
    item_correlation = random.randint(0,2)
    capacity_factor = random.randint(1,10)

    problem = TTP(num_nodes=num_nodes, num_items_per_city=num_items_per_city, item_correlation=item_correlation, capacity_factor=capacity_factor)
    data_root = "data_full" 
    data_dir = pathlib.Path(".")/data_root/"training"/"sop"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "nn_"+str(num_nodes)+"_nipc_"+str(num_items_per_city)+"_"+str(prob_idx)
    dataset_path = data_dir/(dataset_name+".pt")

    with open(dataset_path.absolute(), "wb") as handle:
        pickle.dump(problem, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run(num_samples):
    num_nodes = 50
    num_items_per_city_list = [1,3,5]

    args = [(num_nodes, nic, idx) for nic in num_items_per_city_list for idx in range(num_samples)]
    with Pool(processes=6) as pool:
        L = pool.starmap(generate,args)

if __name__ == "__main__":
    run(1000)

    