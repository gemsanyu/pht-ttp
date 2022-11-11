import pathlib
import os
from copy import deepcopy
from typing import NamedTuple

import torch

from .utils import get_renting_rate, normalize, normalize_coords

CPU_DEVICE = torch.device('cpu')

# knapsack problem profit vs weight parameters
UNCORRELATED=0
CORRELATED_SAME_WEIGHTS=1
CORRELATED=2

# other parameters which are made constant in the ttp benchmark dataset
# luckily the rent ratio is not used because we are solving MO-TTP
MAX_V = 1.
MIN_V = 0.1
CAPACITY_CONSTANT = 11.


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


class TTP(object):
    """
        The Travelling Thief Problem Class
        if file_path is not None, then we're going to read
        the file into dataset,
        else generate training problem based on the given properties
    """
    
    def __init__(self, 
                 num_nodes=10, 
                 num_items_per_city=1, 
                 item_correlation=0, 
                 capacity_factor=2,
                 max_v=MAX_V,
                 min_v=MIN_V,
                 device=CPU_DEVICE,
                 dataset_name=None):
        
        super(TTP, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.num_items_per_city = num_items_per_city
        self.item_correlation = item_correlation
        self.capacity_factor = capacity_factor
        self.max_v = torch.tensor(max_v, device=device).double().to(device)
        self.min_v = torch.tensor(min_v, device=device).double().to(device)
        self.dataset_name = dataset_name

        self.location_data = None
        self.profit_data = None
        self.weight_data = None
        self.max_cap = 0.
        self.num_items = None
        self.item_city_idx = None
        self.item_city_mask = None
        # nondominated archives that save current nondom
        # found in each iteration
        # help improve accuracy of computing HV, or nondom ranking score
        self.nondom_archive = None
        self.reference_point = torch.zeros((1,2), dtype=torch.float32)
        
        if dataset_name is not None:
            dataset_dir = pathlib.Path(".")/"data_full"/"test"
            self.read_dataset_from_file(dataset_dir, dataset_name)
        else:
            self.generate_problem()
        self.density = self.profit_data.norm_profits/self.weight_data.norm_weights
        self.max_profit = torch.sum(self.profit_data.profits, dim=0)
        self.max_travel_time = get_max_travel_time(self.num_nodes, self.location_data.W, 
                                                    self.min_v, self.device)
        
        # self.raw_node_static_features = self.get_raw_node_static_features()
        # self.raw_item_static_features = self.get_raw_item_static_features()
        self.max_travel_time=-1
        
    def generate_problem(self):
        """
        generate the graph then the items
        item_correlation : correlation between item weights and profits
        capacity_factor : the cap will be capacity_factor*sum(weights)
        """
        coords, W = generate_graph(self.num_nodes, device=self.device)
        # norm_coords, norm_W, distance_scale = coords, W, 1 # no need to normalize
        norm_coords, norm_W, distance_scale = normalize_coords(coords, W)
        self.location_data = LocationData(coords, W, norm_coords, norm_W, distance_scale)
        profits, weights, item_city_idx = generate_items(self.num_nodes,
                                                         self.num_items_per_city,
                                                         self.item_correlation,
                                                         self.device)
        total_weights = torch.sum(weights, dim=0)
        self.max_cap = total_weights*self.capacity_factor/CAPACITY_CONSTANT
        self.max_cap = torch.ceil(self.max_cap).double()
        self.max_cap.to(self.device)

        norm_profits, profit_scale = normalize(profits)
        norm_weights, weight_scale = normalize(weights)
        self.profit_data = ProfitData(profits, norm_profits, profit_scale)
        self.weight_data = WeightData(weights, norm_weights, weight_scale)
        self.item_city_idx = item_city_idx
        self.num_items = (self.num_nodes-1)*self.num_items_per_city

        self.item_city_mask = torch.arange(self.num_nodes, device=self.device).expand(self.num_items, self.num_nodes).transpose(1, 0)
        self.item_city_mask = self.item_city_mask == self.item_city_idx.unsqueeze(0)
        self.item_city_mask = self.item_city_mask.bool()
        self.min_tour_length, self.max_profit, self.renting_rate = get_renting_rate(W, weights, profits, self.max_cap)
        self.min_tour_length = torch.tensor(self.min_tour_length, dtype=torch.float32)
        self.max_profit = torch.tensor(self.max_profit, dtype=torch.float32)        


    def read_dataset_from_file(self, dataset_dir, dataset_name):
        data_path = dataset_dir/(dataset_name+".txt")
        coords = None
        weights = None
        profits = None
        item_city_idx = None
        self.batch_size = 1
        with open(data_path.absolute(), "r") as data_file:
            lines = data_file.readlines()
            for i, line in enumerate(lines):
                strings = line.split()
                if i < 2:
                    continue
                elif i == 2:
                    self.num_nodes = int(strings[1])
                    coords = torch.zeros((self.num_nodes, 2), 
                                         dtype=torch.float32,
                                         device=self.device)
                elif i == 3:
                    self.num_items = int(strings[3])
                    weights = torch.zeros(size=(self.num_items,),
                                          dtype=torch.float32,
                                          device=self.device)
                    profits = torch.zeros(size=(self.num_items,),
                                          dtype=torch.float32,
                                          device=self.device)
                    item_city_idx = torch.zeros(size=(self.num_items,),
                                          dtype=torch.long,
                                          device=self.device)
                elif i == 4:
                    self.max_cap = float(strings[3])
                    self.max_cap = torch.tensor(self.max_cap)
                elif i == 5:
                    self.min_v = float(strings[2])
                elif i == 6:
                    self.max_v = float(strings[2])
                elif i == 7:
                    self.renting_rate = float(strings[2])
                elif i < 10:
                    continue
                elif i < self.num_nodes + 10:
                    j = i-10
                    coords[j, 0] = float(strings[1])
                    coords[j, 1] = float(strings[2])
                elif i > self.num_nodes + 10:
                    j = i-self.num_nodes-11
                    profits[j] = float(strings[1])
                    weights[j] = float(strings[2])
                    item_city_idx[j] = int(strings[3]) - 1

        # calculate distance matrix
        W = torch.cdist(coords.double(), coords.double(), p=2).to(self.device)
        W = torch.ceil(W).double()

        norm_coords, norm_W, distance_scale = normalize_coords(coords, W)
        norm_profits, profit_scale = normalize(profits)
        norm_weights, weight_scale = normalize(weights)
        self.location_data = LocationData(coords, W, norm_coords, norm_W, distance_scale)
        self.profit_data = ProfitData(profits, norm_profits, profit_scale)
        self.weight_data = WeightData(weights, norm_weights, weight_scale)                
        self.item_city_idx = item_city_idx
        
        self.item_city_mask = torch.arange(self.num_nodes, device=self.device).expand(self.num_items, self.num_nodes).transpose(1, 0)
        self.item_city_mask = self.item_city_mask == self.item_city_idx.unsqueeze(0)
        self.item_city_mask = self.item_city_mask.bool()

        # read sample solutions/points, if exist
        # this is especially for visualizing the progress
        # in training, otherwise unneeded
        # self.min_tour_length, self.max_profit, self.renting_rate = get_renting_rate(W, weights, profits, self.max_cap)
        self.min_tour_length = torch.tensor(0, dtype=torch.float32)
        self.max_profit = torch.tensor(0, dtype=torch.float32)        
        solution_path = dataset_dir/"solutions"/(dataset_name+".txt")
        if os.path.isfile(solution_path.absolute()):
            solutions = []
            with open(solution_path.absolute(), "r") as data_file:
                lines = data_file.readlines()
                for i, line in enumerate(lines):
                    strings = line.split()
                    sol = [float(strings[0]), float(strings[1])]
                    solutions += [sol]
            self.sample_solutions = torch.tensor(solutions, device=CPU_DEVICE)
            
            
    def get_total_time(self, node_order, item_selection):    
        # get travelled distance list
        max_cap = self.max_cap
        node_order_ = torch.roll(node_order, shifts=-1)
        W = self.location_data.W
        weights = self.weight_data.weights.double()
        selected_weights = weights * item_selection
        selected_weights = torch.cat((torch.tensor([0], device=self.device), selected_weights))
        ordered_selected_weights = selected_weights[node_order]
        ordered_selected_weights = torch.cumsum(ordered_selected_weights, dim=0)
        velocity_range = self.max_v - self.min_v
        velocity_list = self.max_v - (ordered_selected_weights/max_cap)*(velocity_range)
        velocity_list[velocity_list<self.min_v] = self.min_v
        distance_list = W[node_order, node_order_]
        travel_time = distance_list/velocity_list
        travel_time = torch.sum(travel_time, dim=0)
        return travel_time

    def get_total_profit(self, item_selection):
        profits = self.profit_data.profits
        total_profit = torch.sum(profits*item_selection, dim=0)
        return total_profit.double()
    
    def get_total_profit_sequential(self, item_selection):
        profits = self.profit_data.profits
        total_profit = torch.zeros(size=(1,1), dtype=torch.float32).squeeze(0)
        for i in range(self.num_items):
            if item_selection[0, i]:
                total_profit += profits[0, i]
        return total_profit

    def get_raw_node_static_features(self):
        # x, y, average_density in node
        num_features = 3
        features = torch.zeros(size=(self.num_nodes, num_features), dtype=torch.float32, device=self.device)
        # coords
        norm_coords = self.location_data.norm_coords
        features[:, 0] = norm_coords[:, 0]
        features[:, 1] = norm_coords[:, 1]

        # average of density
        node_average_density = torch.sum(self.density.unsqueeze(0) * self.item_city_mask.float(), dim=1) / torch.sum(self.item_city_mask, dim=1)
        node_average_density[node_average_density.isnan()] = 0.
        features[:, 2] = node_average_density
        return features

    def get_raw_item_static_features(self):
        # profits, weights, density
        num_features = 3
        features = torch.zeros(size=(self.num_items, num_features), dtype=torch.float32,device=self.device)
        features[:, 0] = self.profit_data.norm_profits
        features[:, 1] = self.weight_data.norm_weights
        features[:, 2] = self.density
        return features

    def get_total_time_sequential(self, node_order, item_selection):
        travel_time = torch.tensor([0], device=self.device).double()
        current_weight = 0.
        W = self.location_data.W[0]
        weights = self.weight_data.weights[0]
        item_city_idx = self.item_city_idx
        for i in range(self.num_nodes):
            current_node = node_order[i]
            next_node = node_order[(i+1)%self.num_nodes]
            # calculate weights
            for j, city_idx in enumerate(item_city_idx[0]):
                if city_idx == current_node and item_selection[j]:
                    current_weight = current_weight + weights[j]
            distance = W[current_node, next_node]
            current_velocity = self.max_v - (current_weight/self.max_cap)*(self.max_v-self.min_v)
            travel_time = travel_time + distance/current_velocity

        return travel_time

def generate_graph(num_nodes, device=CPU_DEVICE):
    """
        just random [0,1],
        no more clusters
    """
    coords = torch.randint(low=-100, high=100, size=(num_nodes, 2), dtype=torch.float32, device=device)
    W = torch.cdist(coords, coords, p=2)
    # ceiling
    W = torch.ceil(W)
    return coords, W

def generate_items(num_nodes, num_items_per_city, item_correlation, device=CPU_DEVICE):
    """
        Remember, node 0 dont have items
    """
    item_size = (num_items_per_city*(num_nodes-1),)
    if item_correlation == UNCORRELATED:
        profits = torch.randint(low=1, high=1000, size=item_size, device=device).float()
        weights = torch.randint(low=1, high=1000, size=item_size, device=device).float()
    elif item_correlation == CORRELATED:
        weights = torch.randint(low=1, high=1000, size=item_size, device=device).float()
        profits = weights + 100.
    else:
        profits = torch.randint(low=1, high=1000, size=item_size, device=device).float()
        weights = torch.randint(low=1000, high=1010, size=item_size, device=device).float()

    # repeated arange(num_items_per_city) per batch
    item_city_idx = torch.arange(num_nodes-1, device=device) + 1
    item_city_idx = item_city_idx.repeat(num_items_per_city)
    item_city_idx = item_city_idx.expand((num_nodes-1)*num_items_per_city,)
    return profits, weights, item_city_idx

def get_max_travel_time(num_nodes, W, min_v, device=CPU_DEVICE):
    min_val = -99999999
    W_tmp = deepcopy(W)

    # set dist to node 0, and diagonal to maxval
    # to prevent other node to go to node 0 or itself
    diag_idx = torch.arange(num_nodes, device=device)
    W_tmp[diag_idx, diag_idx] = min_val
    W_tmp[:, 0] = min_val

    current_node = 0
    travel_distance = 0.
    for _ in range(num_nodes-1):
        next_node = W_tmp[current_node, :].argmax(dim=0)
        travel_distance = travel_distance + W[current_node, next_node]
        current_node = next_node
        W_tmp[:, next_node] = min_val

    travel_distance = travel_distance + W[current_node, 0]
    max_travel_time = travel_distance/min_v
    return max_travel_time

if __name__=='__main__': 
    A = TTP()