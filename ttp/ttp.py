from copy import deepcopy
import os
import pathlib
from typing import Optional

import torch

from ttp.utils import get_renting_rate, normalize, normalize_coords, read_data
from ttp.utils import LocationData, ProfitData, WeightData
from ttp.utils import generate_item_city_idx, generate_item_city_mask

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
                 dataseed=None,
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
        self.dataseed = dataseed

        self.location_data = Optional[LocationData]
        self.profit_data = Optional[ProfitData]
        self.weight_data = Optional[WeightData]
        self.max_cap = 0.
        self.num_items = None
        self.item_city_idx = None
        self.item_city_mask = None
        # nondominated archives that save current nondom
        # found in each iteration
        # help improve accuracy of computing HV, or nondom ranking score
        self.nondom_archive = None
        self.reference_point = torch.zeros((1,2), dtype=torch.float32)
        self.dataset_dir = pathlib.Path(".")/"data_full"/"test"
        if dataset_name is not None:
            self.init_dataset_from_file(dataset_name)
        else:
            self.generate_problem(self.dataseed)
        self.density = self.profit_data.norm_profits/self.weight_data.norm_weights
        self.max_profit = torch.sum(self.profit_data.profits, dim=0)
        self.max_travel_time = get_max_travel_time(self.num_nodes, self.location_data.W, 
                                                    self.min_v, self.device)
        self.max_travel_time=-1
        
    def generate_problem(self, dataseed=None):
        """
        generate the graph then the items
        item_correlation : correlation between item weights and profits
        capacity_factor : the cap will be capacity_factor*sum(weights)
        """
        dataseed_path = None
        if dataseed is not None:
            dataseed_path = self.dataset_dir/(dataseed+".txt")
        coords, W = generate_graph(self.num_nodes, dataseed_path=dataseed_path, device=self.device)
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

        self.item_city_mask = generate_item_city_mask(self.num_nodes, self.num_items, self.item_city_idx)
        # self.min_tour_length, self.max_profit, self.renting_rate = 0,0,0
        self.min_tour_length, self.max_profit, self.renting_rate = get_renting_rate(W, weights, profits, self.max_cap)
        self.min_tour_length = torch.tensor(self.min_tour_length, dtype=torch.float32)
        self.max_profit = torch.tensor(self.max_profit, dtype=torch.float32)        


    def init_dataset_from_file(self, dataset_name):
        data_path = self.dataset_dir/(dataset_name+".txt")
        self.location_data, self.profit_data, self.weight_data, self.item_city_idx, self.num_nodes, self.num_items, self.renting_rate, self.min_v, self.max_v, self.max_cap = read_data(data_path)
    
        self.item_city_mask = torch.arange(self.num_nodes, device=self.device).expand(self.num_items, self.num_nodes).transpose(1, 0)
        self.item_city_mask = self.item_city_mask == self.item_city_idx.unsqueeze(0)
        self.item_city_mask = self.item_city_mask.bool()

        # read sample solutions/points, if exist
        # this is especially for visualizing the progress
        # in training, otherwise unneeded
        # self.min_tour_length, self.max_profit, self.renting_rate = get_renting_rate(W, weights, profits, self.max_cap)
        self.min_tour_length = torch.tensor(0, dtype=torch.float32)
        self.max_profit = torch.tensor(0, dtype=torch.float32)        
        solution_path = self.dataset_dir/"solutions"/(dataset_name+".txt")
        if os.path.isfile(solution_path.absolute()):
            solutions = []
            with open(solution_path.absolute(), "r") as data_file:
                lines = data_file.readlines()
                for i, line in enumerate(lines):
                    strings = line.split()
                    sol = [float(strings[0]), float(strings[1])]
                    solutions += [sol]
            self.sample_solutions = torch.tensor(solutions, device=CPU_DEVICE)
        else:
            self.sample_solutions = None
            
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

def generate_graph(num_nodes, dataseed_path=None, device=CPU_DEVICE):
    """
        either generate random coords (uniform)
        or
        sample from given coords (from a dataset/dataseed just to differentiate)
    """
    if dataseed_path is None:
        coords = torch.randint(low=-100, high=100, size=(num_nodes, 2), dtype=torch.float32, device=device)
        W = torch.cdist(coords, coords, p=2)
        # ceiling
        W = torch.ceil(W)
    else:
        # sample the dataseed coords for num_nodes of coords
        location_data, _, _, _, _, _, _, _, _, _ = read_data(dataseed_path)
        coords_all, W_all = location_data.coords, location_data.W
        rand_idx = torch.randint(0, len(coords_all), (num_nodes,))
        coords = coords_all[rand_idx, :]
        W = W_all[rand_idx, :][:, rand_idx]

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
    item_city_idx = generate_item_city_idx(num_nodes, num_items_per_city)
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