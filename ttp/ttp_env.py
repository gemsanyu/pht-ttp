from typing import Optional, Tuple
from matplotlib.pyplot import axis
import torch
import numpy as np

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
                 item_city_mask):
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
        self.max_travel_time = 0
        
        # prepare features with dummy items padded
        # add #num_nodes dummy item to the features
        batch_idx = torch.arange(self.batch_size).view(self.batch_size, 1)
        self.item_batch_idx = batch_idx.repeat_interleave(self.num_items).numpy()
        self.node_batch_idx = batch_idx.repeat_interleave(self.num_nodes).numpy()
        self.batch_idx = batch_idx.squeeze(1).numpy()
        self.batch_idx_W = np.repeat(np.arange(self.batch_size)[:,np.newaxis], self.num_nodes, axis=1)

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

    def reset(self):
        self.current_location = np.zeros((self.batch_size,), dtype=np.int64)
        self.current_load = np.zeros((self.batch_size,))
        self.item_selection = np.zeros((self.batch_size, self.num_items), dtype=np.bool)
        self.tour_list = np.zeros((self.batch_size, self.num_nodes), dtype=np.int64)
        self.num_visited_nodes = np.ones((self.batch_size,), dtype=np.int64)
        self.is_selected = np.zeros((self.batch_size, self.num_items+self.num_nodes))
        self.is_node_visited = np.zeros((self.batch_size, self.num_nodes), dtype=np.bool)
        self.is_node_visited[:, 0] = True
        # the dummy item for first city is prohibited until all nodes are visited
        self.eligibility_mask = np.ones((self.batch_size, self.num_items+self.num_nodes), dtype=np.bool)
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
        
        # dist_to_origin, weight, profit, density  
    def get_static_features(self) -> torch.Tensor:
        num_static_features = 4
        static_features = np.zeros((self.batch_size, self.num_items, num_static_features), dtype=np.float32)
        origin_coords = np.expand_dims(self.norm_coords[:,0,:],axis=1)
        item_coords = np.take_along_axis(self.norm_coords, self.item_city_idx[:,:,np.newaxis], 1) 
        item_dist_to_origin = np.linalg.norm(item_coords-origin_coords, axis=2)
        static_features[:, :, 0] = item_dist_to_origin
        static_features[:, :, 1] = self.norm_weights
        static_features[:, :, 2] = self.norm_profits
        static_features[:, :, 3] = self.norm_profits/self.norm_weights

        dummy_static_features = np.zeros((self.batch_size, self.num_nodes, num_static_features), dtype=np.float32)
        dummy_static_features[:,:,0] = np.linalg.norm(origin_coords-self.norm_coords, axis=2)
        static_features = np.concatenate((static_features, dummy_static_features), axis=1)
        return static_features

        # dist_to_curr, current_weight, current_velocity
    def get_dynamic_features(self) -> torch.Tensor:
        num_dynamic_features = 3
        # per item features = distance
        current_coords = np.take_along_axis(self.norm_coords, self.current_location[:,np.newaxis,np.newaxis], 1)
        item_coords = np.take_along_axis(self.norm_coords, self.item_city_idx[:,:,np.newaxis], 1) 
        item_dist_to_curr = np.linalg.norm(current_coords-item_coords, axis=2)
        dummy_dist_to_curr = np.linalg.norm(self.norm_coords-current_coords, axis=2)
        dist_to_curr = np.concatenate((item_dist_to_curr, dummy_dist_to_curr), axis=1)
        dist_to_curr = dist_to_curr[:,:,np.newaxis]
        # global features weigh and velocity
        global_dynamic_features = np.zeros((self.batch_size, 2), dtype=np.float32)
        global_dynamic_features[:, 0] = np.sum(self.norm_weights*self.item_selection, axis=1)
        current_vel = self.max_v - (self.current_load/self.max_cap)*(self.max_v-self.min_v)
        current_vel = np.maximum(current_vel, self.min_v)
        global_dynamic_features[:, 1] = current_vel
        global_dynamic_features = global_dynamic_features[:,np.newaxis,:]
        global_dynamic_features = np.repeat(global_dynamic_features, self.num_items+self.num_nodes, axis=1)
        dynamic_features = np.concatenate([dist_to_curr, global_dynamic_features], axis=2)
        return dynamic_features

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
        items_more_than_cap = np.zeros_like(self.eligibility_mask, dtype=np.bool)
        items_more_than_cap[:, :self.num_items] = self.weights > current_cap[:, np.newaxis]
        self.eligibility_mask = np.logical_and(self.eligibility_mask, ~items_more_than_cap)
        
        # check if the selected item's location is not the current location too
        selected_item_location = self.item_city_idx[active_idx, selected_item]
        is_diff_location = self.current_location[active_idx] != selected_item_location
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
        selected_weights = self.item_selection.astype(dtype=np.float32)*self.weights
        selected_node_weights = np.zeros((self.batch_size, self.num_nodes), dtype=np.float32)
        np.put_along_axis(selected_node_weights, self.item_city_idx, selected_weights, axis=1)
        ordered_selected_weights = selected_node_weights[self.batch_idx_W, self.tour_list]
        final_weights = np.cumsum(ordered_selected_weights, axis=1)
        velocity_list = self.max_v[:,np.newaxis] - (final_weights/self.max_cap[:,np.newaxis])*(self.max_v-self.min_v)[:, np.newaxis]
        # velocity_list[velocity_list<self.min_v.unsqueeze(1)] = self.min_v.unsqueeze(1)
        tour_lengths = (edge_lengths/velocity_list).sum(axis=-1)

        # total profit
        selected_profits = self.item_selection*profits
        total_profits = selected_profits.sum(axis=-1)

        total_cost = total_profits - tour_lengths*self.renting_rate
        return torch.from_numpy(self.tour_list), torch.from_numpy(self.item_selection), torch.from_numpy(tour_lengths), torch.from_numpy(total_profits), torch.from_numpy(total_cost)