import pathlib
import pickle
import platform
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
                 num_nodes:int=30,
                 num_items_per_city:int = 1,
                 item_correlation_list:List[int]=[0,1,2],
                 capacity_factor_list:List[int]=[1,2,3,4,5,6,7,8,9,10],
                 mode="training",
                 dataset_name=None
            ) -> None:
        super(TTPDataset, self).__init__()
        plt = platform.system()
        if plt == 'Linux':
            pathlib.WindowsPath = pathlib.PosixPath
        self.mode = mode
        self.batch = None
        if dataset_name is None:
            self.num_samples = num_samples
            self.num_nodes = num_nodes
            self.num_items_per_city = num_items_per_city
            self.item_correlation_list = item_correlation_list
            self.capacity_factor_list = capacity_factor_list
            self.config_list = [(num_nodes, num_items_per_city, ic, cf) for ic in self.item_correlation_list for cf in capacity_factor_list]
            self.num_configs = len(self.config_list)
            self.batch = None
            self.prob = None
            # we pregenerate some values
            # becuz we wanna add dummy items to 
            # each batch
            # so all have same dimensions
            self.batch_list = []
            for index in range(self.num_samples):
                config_idx = index % len(self.config_list)
                prob_idx = index // len(self.config_list) 
                nn, nipc, ic, cf = self.config_list[config_idx]
                prob = read_prob(self.mode, nn, nipc, ic, cf, prob_idx)
                batch = get_batch_from_prob(prob)
                self.batch_list += [batch]
        else:
            self.num_samples = 2
            self.dataset_path = dataset_name
            prob = TTP(dataset_name=dataset_name)
            self.batch = get_batch_from_prob(prob)
            self.prob = prob

            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.batch is None:
            return self.batch_list[index]
        return self.batch
    
def get_batch_from_prob(prob:TTP):
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


def read_prob(mode="training",num_nodes=None, num_items_per_city=None, item_correlation=None, capacity_factor=None, prob_idx=None, dataset_path=None) -> TTP:
    if dataset_path is None:
        data_root = "data_full" 
        data_dir = pathlib.Path(".")/data_root/mode
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = "nn_"+str(num_nodes)+"_nipc_"+str(num_items_per_city)+"_ic_"+str(item_correlation)+"_cf_"+str(capacity_factor)+"_"+str(prob_idx)
        dataset_path = data_dir/(dataset_name+".pt")
    with open(dataset_path.absolute(), 'rb') as handle:
        prob = pickle.load(handle)
    return prob

def get_dataset_list(num_samples:int=1000000,
                     num_nodes_list:List[int]=[20,30],
                     num_items_per_city_list:List[int] = [1,3,5],
                     item_correlation_list:List[int]=[0,1,2],
                     capacity_factor_list:List[int]=[1,2,3,4,5,6,7,8,9,10],
                     mode="training",):
    dataset_list = [TTPDataset(num_samples, num_nodes=nn, num_items_per_city=nipc, item_correlation_list=item_correlation_list, capacity_factor_list=capacity_factor_list, mode=mode) for nn in num_nodes_list for nipc in num_items_per_city_list]
    return dataset_list

def combine_batch_list(batch_list):
    coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp = batch_list[0]
    for i in range(1,len(batch_list)):
        coordsi, norm_coordsi, Wi, norm_Wi, profitsi, norm_profitsi, weightsi, norm_weightsi, min_vi, max_vi, max_capi, renting_ratei, item_city_idxi, item_city_maski, best_profit_kpi, best_route_length_tspi = batch_list[i]
        coords = torch.cat([coords, coordsi], dim=0)
        norm_coords = torch.cat([norm_coords, norm_coordsi], dim=0)
        W = torch.cat([W, Wi], dim=0)
        norm_W = torch.cat([norm_W, norm_Wi], dim=0)
        profits = torch.cat([profits, profitsi], dim=0)
        norm_profits = torch.cat([norm_profits, norm_profitsi], dim=0)
        weights = torch.cat([weights, weightsi], dim=0)
        norm_weights = torch.cat([norm_weights, norm_weightsi], dim=0)
        min_v = torch.cat([min_v, min_vi], dim=0)
        max_v = torch.cat([max_v, max_vi], dim=0)
        max_cap = torch.cat([max_cap, max_capi], dim=0)
        renting_rate = torch.cat([renting_rate, renting_ratei], dim=0)
        item_city_idx = torch.cat([item_city_idx, item_city_idxi], dim=0)
        item_city_mask = torch.cat([item_city_mask, item_city_maski], dim=0)
        best_profit_kp = torch.cat([best_profit_kp, best_profit_kpi], dim=0)
        best_route_length_tsp = torch.cat([best_route_length_tsp, best_route_length_tspi], dim=0)
    return coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, best_profit_kp, best_route_length_tsp


if __name__ == "__main__":
    train_dataset = TTPDataset(1000000)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for batch_idx, batch in enumerate(train_dataloader):
        coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, renting_rate, item_city_idx, item_city_mask = batch
        