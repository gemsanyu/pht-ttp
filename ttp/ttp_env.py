from typing import Optional, Tuple
from matplotlib.pyplot import axis
import torch
import numba as nb
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances




@nb.njit((nb.float32[:,:])(nb.float32[:,:], nb.float32[:,:,:], nb.int64[:], nb.int64[:,:], nb.float64[:], nb.int64, nb.int64, nb.int64), cache=True)
def get_trav_time_to_curr(out, 
                          distance_matrix, 
                          current_location, 
                          item_city_idx,
                          current_vel,
                          batch_size,
                          num_items,
                          num_nodes):
    for bi in range(batch_size):
        curr = current_location[bi]
        c_vel = current_vel[bi]
        dm_temp = distance_matrix[bi,:,curr]
        ic_temp = item_city_idx[bi,:]
        for i in range(num_items):
            out[bi, i] = dm_temp[ic_temp[i]]/c_vel
        for i in range(num_nodes):
            out[bi,num_items+i] = dm_temp[i]/c_vel
    return out

@nb.njit((nb.float32[:,:,:])(nb.float32[:,:,:], nb.float32[:,:], nb.boolean[:,:], nb.float64[:], nb.int64, nb.int64), cache=True)
def get_global_dynamic_features(out, 
                                norm_weights, 
                                item_selection, 
                                current_vel,
                                batch_size,
                                num_items):
    for bi in range(batch_size):
        sum_weights:float = 0
        for i in range(num_items):
            sum_weights += norm_weights[bi,i]*item_selection[bi,i]
        out[bi,:,0] = sum_weights
        out[bi,:,1] = current_vel[bi]
    return out

class TTPEnv():
    def __init__(self,
                 coords, 
                 norm_coords, 
                 W, 
                 norm_W, 
                 profits, 
                 norm_profits, 
                 weights, 
                 norm_weights, 
                 min_v, 
                 max_v, 
                 max_cap,
                 renting_rate, 
                 item_city_idx,
                 item_city_mask,
                 best_profit_kp,
                 best_route_length_tsp):
        self.batch_size, self.num_nodes, _ = coords.shape
        _, self.num_items = profits.shape
        self.coords = coords.numpy()
        self.norm_coords = norm_coords.numpy()
        self.W = W.numpy()
        self.norm_W = norm_W.numpy()
        self.profits = profits.numpy()
        self.max_profit = profits.sum(-1).numpy()
        self.norm_profits = norm_profits.numpy()
        self.weights = weights.numpy()
        self.norm_weights = norm_weights.numpy()
        self.min_v = min_v.numpy()
        self.max_v = max_v.numpy()
        self.max_cap = max_cap.numpy()
        self.renting_rate = renting_rate.numpy()
        self.item_city_idx = item_city_idx.numpy()
        self.item_city_mask = item_city_mask.numpy()
        self.best_profit_kp = best_profit_kp.numpy()
        self.best_route_length_tsp = best_route_length_tsp.numpy()
        self.max_travel_time = 0
        
        # prepare features with dummy items padded
        # add #num_nodes dummy item to the features
        batch_idx = torch.arange(self.batch_size).view(self.batch_size, 1)
        self.item_batch_idx = batch_idx.repeat_interleave(self.num_items).numpy()
        self.node_batch_idx = batch_idx.repeat_interleave(self.num_nodes).numpy()
        self.batch_idx = batch_idx.squeeze(1).numpy()
        self.batch_idx_W = np.repeat(np.arange(self.batch_size)[:,np.newaxis], self.num_nodes, axis=1)
        item_coords = np.take_along_axis(self.norm_coords, self.item_city_idx[:,:,np.newaxis], 1) 
        origin_coords = np.expand_dims(self.norm_coords[:,0,:],axis=1)
        item_dist_to_origin = np.linalg.norm(item_coords-origin_coords, axis=2)
        dummy_dist_to_origin = np.linalg.norm(self.norm_coords-origin_coords, axis=2)
        dist_to_origin = np.concatenate((item_dist_to_origin, dummy_dist_to_origin), axis=1)
        self.dist_to_origin = dist_to_origin[:,:,np.newaxis]
        self.distance_matrix = np.concatenate([euclidean_distances(norm_coords[i,:,:])[np.newaxis,:,:] for i in range(self.batch_size)])
        
        self.current_location = None
        self.current_load = None
        self.item_selection = None
        self.tour_list = None
        self.num_visited_nodes = None
        self.is_selected = None
        self.is_node_visited = None
        self.eligibility_mask = None
        self.all_city_idx = None
        self.static_features = None
        self.dynamic_features = np.zeros((self.batch_size, self.num_items+self.num_nodes,4), dtype=np.float32)
        

    def reset(self):
        self.current_location = np.zeros((self.batch_size,), dtype=np.int64)
        self.current_load = np.zeros((self.batch_size,))
        self.item_selection = np.zeros((self.batch_size, self.num_items), dtype=bool)
        self.tour_list = np.zeros((self.batch_size, self.num_nodes), dtype=np.int64)
        self.num_visited_nodes = np.ones((self.batch_size,), dtype=np.int64)
        self.is_selected = np.zeros((self.batch_size, self.num_items+self.num_nodes))
        self.is_node_visited = np.zeros((self.batch_size, self.num_nodes), dtype=bool)
        self.is_node_visited[:, 0] = True
        # the dummy item for first city is prohibited until all nodes are visited
        self.eligibility_mask = np.ones((self.batch_size, self.num_items+self.num_nodes), dtype=bool)
        self.eligibility_mask[:, self.num_items] = False
        dummy_idx = torch.arange(self.num_nodes, dtype=torch.long)
        dummy_idx = dummy_idx.unsqueeze(0).expand(self.batch_size, self.num_nodes).numpy()
        self.all_city_idx = np.concatenate((self.item_city_idx, dummy_idx),axis=1)
        self.static_features = self.get_static_features()
        
    def begin(self):
        self.reset()
        dynamic_features = self.get_dynamic_features()
        eligibility_mask = self.eligibility_mask
        return self.static_features, dynamic_features, eligibility_mask
        
        # weight, profit, density  
    def get_static_features(self) -> torch.Tensor:
        num_static_features = 3
        static_features = np.zeros((self.batch_size, self.num_items, num_static_features), dtype=np.float32)
        static_features[:, :, 0] = self.norm_weights
        static_features[:, :, 1] = self.norm_profits
        static_features[:, :, 2] = self.norm_profits/self.norm_weights
        static_features = np.nan_to_num(static_features, nan=0)
        weights_per_city = self.norm_weights[:,np.newaxis,:]
        weights_per_city = np.repeat(weights_per_city, repeats=self.num_nodes, axis=1)
        weights_per_city = weights_per_city*self.item_city_mask
        profits_per_city = self.norm_profits[:,np.newaxis,:]
        profits_per_city = np.repeat(profits_per_city, repeats=self.num_nodes, axis=1)
        profits_per_city = profits_per_city*self.item_city_mask
        with np.errstate(divide="ignore", invalid="ignore"):
            density_per_city = profits_per_city/weights_per_city
        density_per_city = np.nan_to_num(density_per_city, nan=0)
        weights_per_city = np.average(weights_per_city, axis=2, keepdims=True)
        profits_per_city = np.average(profits_per_city, axis=2, keepdims=True)
        density_per_city = np.average(density_per_city, axis=2, keepdims=True)
        # print(self.item_city_mask.shape)
        # print(density_per_city)
        dummy_static_features = np.concatenate([weights_per_city,profits_per_city,density_per_city], axis=2)
        # dummy_static_features = np.zeros((self.batch_size, self.num_nodes, num_static_features), dtype=np.float32)
        # print(dummy_static_features.shape)
        # exit()
        # dummy_static_features[:,:,0] = np.linalg.norm(origin_coords-self.norm_coords, axis=2)
        static_features = np.concatenate((static_features, dummy_static_features), axis=1)
        # static_features = np.nan_to_num(static_features, nan=0)
        return static_features

        # trav_time_to_origin, trav_time_to_curr, current_weight, current_velocity
    def get_dynamic_features(self) -> torch.Tensor:
        current_vel = self.max_v - (self.current_load/self.max_cap)*(self.max_v-self.min_v)
        current_vel = np.maximum(current_vel, self.min_v)
        current_vel = self.max_v - (self.current_load/self.max_cap)*(self.max_v-self.min_v)
        current_vel = np.maximum(current_vel, self.min_v)
        self.dynamic_features[:,:,0] = (self.dist_to_origin/current_vel[:, np.newaxis, np.newaxis])[:,:,0]
        self.dynamic_features[:,:,1] = get_trav_time_to_curr(self.dynamic_features[:,:,1], self.distance_matrix, self.current_location, self.item_city_idx, current_vel, self.batch_size, self.num_items, self.num_nodes)
        
        # global features weigh and velocity
        self.dynamic_features[:,:,2:4] = get_global_dynamic_features(self.dynamic_features[:,:,2:4], self.norm_weights, self.item_selection, current_vel, self.batch_size, self.num_items)
        return self.dynamic_features

    def act(self, active_idx:torch.Tensor, selected_idx:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        # filter which is taking item, which is visiting nodes only
        active_idx = active_idx.cpu().numpy()
        selected_idx = selected_idx.cpu().numpy()
        is_taking_item = selected_idx < self.num_items
        if np.any(is_taking_item):
            self.take_item(active_idx[is_taking_item], selected_idx[is_taking_item])
        is_visiting_node_only = np.logical_not(is_taking_item)
        if np.any(is_visiting_node_only):
            self.visit_node(active_idx[is_visiting_node_only], selected_idx[is_visiting_node_only]-self.num_items)

        dynamic_features = self.get_dynamic_features()
        return dynamic_features, self.eligibility_mask

    def take_item(self, active_idx, selected_item):
        # set item as selected in item selection
        self.is_selected[active_idx, selected_item] = True
        self.item_selection[active_idx, selected_item] = True
        self.eligibility_mask[active_idx, selected_item] = False
        
        # update current weight
        self.current_load[active_idx] += self.weights[active_idx, selected_item]
        current_cap = self.max_cap - self.current_load
        # append dummy zeros for dummy items
        items_more_than_cap = np.zeros_like(self.eligibility_mask, dtype=bool)
        items_more_than_cap[:, :self.num_items] = self.weights > current_cap[:, np.newaxis]
        self.eligibility_mask = np.logical_and(self.eligibility_mask, ~items_more_than_cap)
        
        # check if the selected item's location is not the current location too
        selected_item_location = self.item_city_idx[active_idx, selected_item]
        is_diff_location = self.current_location[active_idx] != selected_item_location
        # print(selected_item)
        if np.any(is_diff_location):
            self.visit_node(active_idx[is_diff_location], selected_item_location[is_diff_location])

    def visit_node(self, active_idx, selected_node):
        # set is selected
        self.is_selected[active_idx, selected_node+self.num_items] = True
        # set dummy item for the selected location is infeasible
        self.eligibility_mask[active_idx, selected_node+self.num_items] = False
        
        # mask all current locations' item as ineligible before moving
        current_locations_item_mask = self.all_city_idx[active_idx, :] == self.current_location[active_idx][:, np.newaxis]
        self.eligibility_mask[active_idx,:] = np.logical_and(self.eligibility_mask[active_idx,:], ~current_locations_item_mask)

        # set current location to next location
        self.current_location[active_idx] = selected_node

        # save it to tour list
        # print(self.num_visited_nodes)
        # print(self.tour_list.shape, active_idx,self.num_visited_nodes[active_idx], selected_node )
        # print(self.eligibility_mask[active_idx])
        # print(self.is_not_dummy_mask[active_idx])
        # print("-----------------------------")
        self.tour_list[active_idx, self.num_visited_nodes[active_idx]] = selected_node
        
        self.num_visited_nodes[active_idx] += 1

        #check if all nodes are visited, if yes then make the dummy item for first city feasibe
        is_all_visited = self.num_visited_nodes == self.num_nodes+1
        if np.any(is_all_visited):
            self.eligibility_mask[is_all_visited, self.num_items] = True

    def finish(self, normalized=False):
        # computing tour lenghts or travel time
        tour_A = self.tour_list
        tour_B = np.roll(tour_A, shift=-1, axis=1)
        if normalized:
            W, profits = self.norm_W, self.norm_profits
        else:
            W, profits = self.W, self.profits

        edge_lengths = W[self.batch_idx_W, tour_A, tour_B]
        selected_weights = self.item_selection.astype(dtype=np.float64)*self.weights
        selected_node_weights = np.zeros((self.batch_size, self.num_nodes), dtype=np.float64)
        np.put_along_axis(selected_node_weights, self.item_city_idx, selected_weights, axis=1)
        ordered_selected_weights = selected_node_weights[self.batch_idx_W, self.tour_list]
        final_weights = np.cumsum(ordered_selected_weights, axis=1)
        velocity_list = self.max_v[:,np.newaxis] - (final_weights/self.max_cap[:,np.newaxis])*(self.max_v-self.min_v)[:, np.newaxis]
        # velocity_list[velocity_list<self.min_v.unsqueeze(1)] = self.min_v.unsqueeze(1)
        tour_lengths = (edge_lengths/velocity_list).sum(axis=-1)

        # total profit
        selected_profits = self.item_selection*profits
        total_profits = selected_profits.sum(axis=-1)
        travel_cost = tour_lengths*self.renting_rate
        total_cost = total_profits - travel_cost
        return self.tour_list, self.item_selection, tour_lengths, total_profits, travel_cost, total_cost