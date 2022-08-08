import pathlib
import pickle
import random
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader

from ttp.ttp import TTP

# knapsack problem profit vs weight parameters
# UNCORRELATED=0
# CORRELATED_SAME_WEIGHTS=1
# CORRELATED=2

# other parameters which are made constant in the ttp benchmark dataset
# MAX_V = 1.
# MIN_V = 0.1
# CAPACITY_CONSTANT = 11.

class TTPDataset(Dataset):
    def __init__(self,
                 num_samples:int=1000000,
                 num_nodes:int = 50,
                 num_items_per_city:int =1,
                 dataset_name=None
            ) -> None:
        super(TTPDataset, self).__init__()

        if dataset_name is None:
            self.num_samples = num_samples
            self.num_nodes = num_nodes
            self.num_items_per_city = num_items_per_city
            self.num_items_idx = 0
            self.prob = None
            self.config_iterator = 0
        else:
            self.num_samples = 2
            self.dataset_path = dataset_name
            self.prob = TTP(dataset_name=dataset_name)
            self.num_nodes = self.prob.num_nodes
            self.num_items_per_city = self.prob.num_items_per_city
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.prob is None:
            prob = read_prob(num_nodes=self.num_nodes, num_items_per_city=self.num_items_per_city, prob_idx=index%1000)
        else:
            prob = self.prob
        coords, norm_coords, W, norm_W = prob.location_data.coords, prob.location_data.norm_coords, prob.location_data.W, prob.location_data.norm_W
        profits, norm_profits = prob.profit_data.profits, prob.profit_data.norm_profits
        weights, norm_weights = prob.weight_data.weights, prob.weight_data.norm_weights
        min_v, max_v, renting_rate = prob.min_v, prob.max_v, prob.renting_rate
        max_cap = prob.max_cap
        item_city_idx, item_city_mask = prob.item_city_idx, prob.item_city_mask
        return coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask


def read_prob(num_nodes, num_items_per_city, prob_idx):
    data_root = "data_full" 
    data_dir = pathlib.Path(".")/data_root/"training"/"sop"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "nn_"+str(num_nodes)+"_nipc_"+str(num_items_per_city)+"_"+str(prob_idx)
    dataset_path = data_dir/(dataset_name+".pt")
    with open(dataset_path.absolute(), 'rb') as handle:
        prob = pickle.load(handle)
    return prob


if __name__ == "__main__":
    train_dataset = TTPDataset(1000000)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for batch_idx, batch in enumerate(train_dataloader):
        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, renting_rate, item_city_idx, item_city_mask = batch
        