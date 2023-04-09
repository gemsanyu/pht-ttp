import pathlib
import pickle
import random
from typing import List
import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader

from ttp.ttp import TTP
from ttp.ttp_env import TTPEnv
from ttp.utils import generate_item_city_idx, generate_item_city_mask

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
                 num_nodes_list:List[int] = [20,30],
                 num_items_per_city_list:List[int] = [1,3,5],
                 item_correlation_list:List[int]=[0,1,2],
                 mode="training",
                 dataset_name=None
            ) -> None:
        super(TTPDataset, self).__init__()

        self.mode = mode
        if dataset_name is None:
            self.num_samples = num_samples
            self.num_nodes_list = num_nodes_list
            self.num_items_per_city_list = num_items_per_city_list
            self.item_correlation_list = item_correlation_list
            self.config_list = [(nn, nipc, ic) for nn in self.num_nodes_list for nipc in self.num_items_per_city_list for ic in self.item_correlation_list]
            self.num_configs = len(self.config_list)
            self.prob = None


            # we pregenerate some values
            # becuz we wanna add dummy items to 
            # each batch
            # so all have same dimensions
            max_num_nodes = max(self.num_nodes_list)
            max_nipc = max(self.num_items_per_city_list)
            max_num_items = max_nipc*(max_num_nodes-1)
            
            self.d_item_city_idx = generate_item_city_idx(max_num_nodes, max_nipc)
            self.d_item_city_mask = generate_item_city_mask(max_num_nodes, max_num_items, self.d_item_city_idx)
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
            config = self.config_list[index%self.num_configs]
            nn, nipc, ic = config
            prob_idx = index//self.num_configs
            if self.mode=="validation":
                prob = read_prob(self.mode,num_nodes=nn, num_items_per_city=nipc, item_correlation=ic, prob_idx=prob_idx)
            else:
                capacity_factor = random.randint(1,10)
                nn = random.choice(self.num_nodes_list)
                nipc = random.choice(self.num_items_per_city_list)
                ic = random.choice(self.item_correlation_list)    
                prob = TTP(num_nodes=nn, num_items_per_city=nipc, item_correlation=ic, capacity_factor=capacity_factor, dataseed="eil76-n75")
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

        num_nodes,_ = coords.shape
        num_items = profits.shape[0]
        if self.prob is not None:
            #all are not dummy
            is_not_dummy_mask = torch.ones((num_nodes+num_items,)).bool()
            return coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp

        # coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights, min_v, max_v, max_cap, renting_rate, item_city_idx, item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp
        max_num_nodes = max(self.num_nodes_list)
        max_nipc = max(self.num_items_per_city_list)
        max_num_items = max_nipc*(max_num_nodes-1)
        data_with_dummy = add_dummy_to_data(max_num_nodes, max_nipc, max_num_items, nipc, coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights)
        d_coords, d_norm_coords, d_W, d_norm_W, d_profits, d_norm_profits, d_weights, d_norm_weights = data_with_dummy
        d_item_city_idx = self.d_item_city_idx.clone()
        d_item_city_mask = self.d_item_city_mask.clone()
        d_idx = torch.arange(max_num_items)
        is_not_dummy_item = d_item_city_idx <= (num_nodes - 1)
        duplicate_idx = d_idx // max_num_nodes
        is_not_dummy_item = torch.logical_and(is_not_dummy_item,(duplicate_idx +1) <= nipc)
        is_not_dummy_nodes = torch.arange(max_num_nodes)<num_nodes
        is_not_dummy_mask = torch.cat([is_not_dummy_item, is_not_dummy_nodes]).bool()
        return d_coords, d_norm_coords, d_W, d_norm_W, d_profits, d_norm_profits, d_weights, d_norm_weights, min_v, max_v, max_cap, renting_rate, d_item_city_idx, d_item_city_mask, is_not_dummy_mask, best_profit_kp, best_route_length_tsp
    

def add_dummy_to_data(max_num_nodes, max_nipc, max_num_items, nipc, coords, norm_coords, W, norm_W, profits, norm_profits, weights, norm_weights):
    num_nodes,_ = coords.shape
    num_dummy_nodes = max_num_nodes-num_nodes
    num_items = profits.shape[0]
    num_dummy_items = max_num_items-num_items 
    d_nipc = max_nipc-nipc 
    
    #coords and W
    c_pad, w_pad = (0,0,0,num_dummy_nodes), (0,num_dummy_nodes,0,num_dummy_nodes)
    d_coords, d_norm_coords = pad(coords,c_pad), pad(norm_coords,c_pad)
    d_W, d_norm_W = pad(W,w_pad), pad(norm_W, w_pad)
    # profits and weights
    pw_pad = (0,num_dummy_items)
    d_profits, d_norm_profits = pad(profits, pw_pad), pad(norm_profits, pw_pad)
    d_weights, d_norm_weights = pad(weights, pw_pad,value=1), pad(norm_weights, pw_pad, value=1)

    # # what is not dummy?
    # # for each non_dummy nodes, num_items items is not dummy
    # non_dummy_node_items_mask = torch.cat([torch.ones((nipc,)),torch.zeros(d_nipc,)])
    # non_dummy_node_items_mask = torch.tile(non_dummy_node_items_mask, (num_nodes-1,))
    # # num_dummy_nodes nodes'items are all dummy
    # dummy_node_items_mask = torch.zeros((max_nipc,))
    # dummy_node_items_mask = torch.tile(dummy_node_items_mask, (num_dummy_nodes,))
    # # num_nodes nodes are not dummy
    # non_dummy_nodes_mask = torch.ones((num_nodes,))
    # dummy_nodes_mask = torch.zeros((num_dummy_nodes,))
    # is_not_dummy_mask = torch.cat([non_dummy_node_items_mask,dummy_node_items_mask,non_dummy_nodes_mask,dummy_nodes_mask]).bool()
    return d_coords, d_norm_coords, d_W, d_norm_W, d_profits, d_norm_profits, d_weights, d_norm_weights


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


def read_prob(mode="training",num_nodes=None, num_items_per_city=None, item_correlation=None, prob_idx=None, dataset_path=None) -> TTP:
    if dataset_path is None:
        data_root = "data_full" 
        data_dir = pathlib.Path(".")/data_root/mode
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = "nn_"+str(num_nodes)+"_nipc_"+str(num_items_per_city)+"_ic_"+str(item_correlation)+"_"+str(prob_idx)
        dataset_path = data_dir/(dataset_name+".pt")
    with open(dataset_path.absolute(), 'rb') as handle:
        prob = pickle.load(handle)
    return prob


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
        