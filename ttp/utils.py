import pathlib
import pickle
from typing import NamedTuple, Union

import torch

from ttp.solver import solve_tsp, solve_knapsack

CPU_DEVICE = torch.device("cpu")


class LocationData(NamedTuple):
    coords: torch.Tensor 
    W: torch.Tensor 
    norm_coords: torch.Tensor 
    norm_W: torch.Tensor 
    distance_scale: torch.Tensor

class ProfitData(NamedTuple):
    profits: torch.Tensor 
    norm_profits: torch.Tensor 
    profit_scale: torch.Tensor 

class WeightData(NamedTuple):
    weights: torch.Tensor 
    norm_weights: torch.Tensor 
    weight_scale: torch.Tensor 


def normalize(values):
    total = torch.sum(values, dim=0)
    norm_values = values/total
    return norm_values, total

def normalize_coords(coords, W=None):
    """
        normalizing coordinates so that in every benchmark dataset
        the range of the coordinates stays the same, while also maintaining
        their relative distances
        normalize to range [0,1]
        by shifting to center, then divide by scale like standard normal dist
    """
    num_nodes, _ = coords.shape

    # get mid and scale, then broadcast
    max_x, _ = torch.max(coords[:, 0], dim=0)
    min_x, _ = torch.min(coords[:, 0], dim=0)
    mid_x = (max_x + min_x)/2.
    mid_x = mid_x.expand(num_nodes)

    max_y, _ = torch.max(coords[:, 1], dim=0)
    min_y, _ = torch.min(coords[:, 1], dim=0)
    mid_y = (max_y + min_y)/2.
    mid_y = mid_y.expand(num_nodes)

    scale_x = max_x - min_x
    scale_y = max_y - min_y
    scale = max(scale_x, scale_y)

    norm_coords = coords.detach().clone()
    norm_coords[:, 0] -= mid_x
    norm_coords[:, 1] -= mid_y
    norm_coords /= scale
    norm_coords += 0.5  # to scale from 0 to 1, else it will scale [-0.5, 0.5]

    if W is None:
        return norm_coords, scale

    # if we also normalized the distance
    # , then just divide them by the scale
    norm_W = W.detach().clone() / scale

    return norm_coords, norm_W, scale


def generate_item_city_idx(num_nodes, num_items_per_city):
    item_city_idx = torch.arange(num_nodes-1) + 1
    item_city_idx = item_city_idx.repeat(num_items_per_city)
    item_city_idx = item_city_idx.expand((num_nodes-1)*num_items_per_city,)
    return item_city_idx

def generate_item_city_mask(num_nodes, num_items, item_city_idx):
    item_city_mask = torch.arange(num_nodes).expand(num_items, num_nodes).transpose(1, 0)
    item_city_mask = item_city_mask == item_city_idx.unsqueeze(0)
    item_city_mask = item_city_mask.bool()
    return item_city_mask.bool()

# get renting rate by solving both knapsack and TSP
def get_renting_rate(W, weights, profits, capacity):
    # solve the knapsack first
    optimal_profit, item_selection = solve_knapsack(weights, profits, capacity)
    # solve the tsp
    route_list, optimal_tour_length = solve_tsp(W)
    renting_rate = float(optimal_profit)/float(optimal_tour_length)
    return optimal_tour_length, optimal_profit, renting_rate

def read_data(data_path, device=CPU_DEVICE) -> Union[LocationData,ProfitData,WeightData,int,int,float,float,float]:
    coords = None
    weights = None
    profits = None
    item_city_idx = None
    num_nodes, num_items = None, None
    renting_rate = None
    min_v, max_v = None, None
    max_cap = None
    with open(data_path.absolute(), "r") as data_file:
        lines = data_file.readlines()
        for i, line in enumerate(lines):
            strings = line.split()
            if i < 2:
                continue
            elif i == 2:
                num_nodes = int(strings[1])
                coords = torch.zeros((num_nodes, 2), 
                                        dtype=torch.float32,
                                        device=device)
            elif i == 3:
                num_items = int(strings[3])
                weights = torch.zeros(size=(num_items,),
                                        dtype=torch.float32,
                                        device=device)
                profits = torch.zeros(size=(num_items,),
                                        dtype=torch.float32,
                                        device=device)
                item_city_idx = torch.zeros(size=(num_items,),
                                        dtype=torch.long,
                                        device=device)
            elif i == 4:
                max_cap = float(strings[3])
                max_cap = torch.tensor(max_cap)
            elif i == 5:
                min_v = float(strings[2])
            elif i == 6:
                max_v = float(strings[2])
            elif i == 7:
                renting_rate = float(strings[2])
            elif i < 10:
                continue
            elif i < num_nodes + 10:
                j = i-10
                coords[j, 0] = float(strings[1])
                coords[j, 1] = float(strings[2])
            elif i > num_nodes + 10:
                j = i-num_nodes-11
                profits[j] = float(strings[1])
                weights[j] = float(strings[2])
                item_city_idx[j] = int(strings[3]) - 1

    # calculate distance matrix
    W = torch.cdist(coords.double(), coords.double(), p=2).to(device)
    W = torch.ceil(W).double()

    norm_coords, norm_W, distance_scale = normalize_coords(coords, W)
    norm_profits, profit_scale = normalize(profits)
    norm_weights, weight_scale = normalize(weights)
    location_data = LocationData(coords, W, norm_coords, norm_W, distance_scale)
    profit_data = ProfitData(profits, norm_profits, profit_scale)
    weight_data = WeightData(weights, norm_weights, weight_scale)                
    item_city_idx = item_city_idx
    return location_data, profit_data, weight_data, item_city_idx, num_nodes, num_items, renting_rate, min_v, max_v, max_cap

def save_prob(problem, num_nodes, num_items_per_city, prob_idx):
    data_root = "data_full"
    data_dir = pathlib.Path(".")/data_root/"training"/"sop"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "nn_"+str(num_nodes)+"_nipc_"+str(num_items_per_city)+"_"+str(prob_idx)
    dataset_path = data_dir/(dataset_name+".pt")

    with open(dataset_path.absolute(), "wb") as handle:
        pickle.dump(problem, handle, protocol=pickle.HIGHEST_PROTOCOL)