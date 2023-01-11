import pathlib
import pickle
import random
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader

from ttp.ttp import TTP
from ttp.ttp_env import TTPEnv

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
        best_profit_kp = prob.max_profit
        best_route_length_tsp = prob.min_tour_length
        return coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp

def prob_list_to_env(prob_list):
    coords_list, norm_coords_list, W_list, norm_W_list = [],[],[],[]
    profits_list, norm_profits_list, weights_list, norm_weights_list = [],[],[],[]
    min_v_list, max_v_list, renting_rate_list, max_cap_list = [],[],[],[]
    item_city_idx_list, item_city_mask_list = [],[]
    best_profit_kp_list, best_route_length_tsp_list = [],[]
    for prob in prob_list:
        coords, norm_coords, W, norm_W = prob.location_data.coords.unsqueeze(0), prob.location_data.norm_coords.unsqueeze(0), prob.location_data.W.unsqueeze(0), prob.location_data.norm_W.unsqueeze(0)
        coords_list += [coords]
        norm_coords_list += [norm_coords]
        W_list += [W]
        norm_W_list += [norm_W] 
        profits, norm_profits = prob.profit_data.profits.unsqueeze(0), prob.profit_data.norm_profits.unsqueeze(0)
        weights, norm_weights = prob.weight_data.weights.unsqueeze(0), prob.weight_data.norm_weights.unsqueeze(0)
        profits_list += [profits]
        norm_profits_list += [norm_profits]
        weights_list += [weights]
        norm_weights_list += [norm_weights]
        min_v, max_v, renting_rate = prob.min_v.unsqueeze(0), prob.max_v.unsqueeze(0), prob.renting_rate
        max_cap = prob.max_cap.unsqueeze(0)
        min_v_list += [min_v]
        max_v_list += [max_v]
        renting_rate_list += [renting_rate]
        max_cap_list += [max_cap]        
        item_city_idx, item_city_mask = prob.item_city_idx.unsqueeze(0), prob.item_city_mask.unsqueeze(0)
        best_profit_kp, best_route_length_tsp = prob.max_profit, prob.min_tour_length
        item_city_idx_list += [item_city_idx]
        item_city_mask_list += [item_city_mask]
        best_profit_kp_list += [best_profit_kp]
        best_route_length_tsp_list += [best_route_length_tsp]
    
    coords = torch.cat(coords_list)
    norm_coords = torch.cat(norm_coords_list)
    W = torch.cat(W_list)
    norm_W = torch.cat(norm_W_list)
    profits =  torch.cat(profits_list)
    norm_profits = torch.cat(norm_profits_list)
    weights = torch.cat(weights_list)
    norm_weights = torch.cat(norm_weights_list)
    min_v = torch.cat(min_v_list)
    max_v = torch.cat(max_v_list)
    max_cap = torch.cat(max_cap_list) 
    renting_rate =  torch.Tensor(renting_rate_list)
    item_city_idx = torch.cat(item_city_idx_list) 
    item_city_mask = torch.cat(item_city_mask_list) 
    best_profit_kp = torch.Tensor(best_profit_kp_list)
    best_route_length_tsp = torch.Tensor(best_route_length_tsp_list)
    env = TTPEnv(coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp)
    return env


def read_prob(num_nodes=None, num_items_per_city=None, prob_idx=None, dataset_path=None) -> TTP:
    if dataset_path is None:
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
        