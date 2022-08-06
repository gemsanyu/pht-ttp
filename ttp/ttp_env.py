from operator import is_not
import torch

class TTPEnv(object):
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
        self.coords = coords
        self.norm_coords = norm_coords
        self.W = W
        self.norm_W = norm_W
        self.profits = profits
        self.max_profit = profits.sum(-1)
        self.norm_profits = norm_profits
        self.weights = weights
        self.norm_weights = norm_weights
        self.min_v = min_v
        self.max_v = max_v
        self.max_cap = max_cap
        self.renting_rate = renting_rate
        self.item_city_idx = item_city_idx
        self.item_city_mask = item_city_mask
        self.max_travel_time = 0
        
        # prepare features with dummy items padded
        # add #num_nodes dummy item to the features
        batch_idx = torch.arange(self.batch_size).view(self.batch_size, 1)
        self.item_batch_idx = batch_idx.repeat_interleave(self.num_items)
        self.node_batch_idx = batch_idx.repeat_interleave(self.num_nodes)
        self.batch_idx = batch_idx.squeeze(1)
        
        self.current_location = None
        self.current_load = None
        self.item_selection = None
        self.tour_list = None
        self.num_visited_nodes = None
        self.is_selected = None
        self.is_node_visited = None
        self.eligibility_mask = None
        self.all_city_idx = None

    def reset(self):
        self.current_location = torch.zeros((self.batch_size,), dtype=torch.long)
        self.current_load = torch.zeros((self.batch_size,))
        self.item_selection = torch.zeros((self.batch_size, self.num_items), dtype=torch.bool)
        self.tour_list = torch.zeros((self.batch_size, self.num_nodes), dtype=torch.long)
        self.num_visited_nodes = torch.ones((self.batch_size,), dtype=torch.long)
        self.is_selected = torch.zeros((self.batch_size, self.num_items+self.num_nodes))
        self.is_node_visited = torch.zeros((self.batch_size, self.num_nodes), dtype=torch.bool)
        self.is_node_visited[:, 0] = True
        # the dummy item for first city is prohibited until all nodes are visited
        self.eligibility_mask = torch.ones((self.batch_size, self.num_items+self.num_nodes), dtype=torch.bool)
        self.eligibility_mask[:, self.num_items] = False
        dummy_idx = torch.arange(self.num_nodes, dtype=torch.long)
        dummy_idx = dummy_idx.unsqueeze(0).expand(self.batch_size, self.num_nodes)
        self.all_city_idx = torch.cat((self.item_city_idx, dummy_idx),dim=1)
        self.static_features = self.get_static_features()
        

    def begin(self):
        self.reset()
        dynamic_features = self.get_dynamic_features()
        eligibility_mask = self.eligibility_mask
        return self.static_features, dynamic_features, eligibility_mask
        
        # dist_to_origin, weight, profit, density  
    def get_static_features(self):
        self.num_static_features = 4
        dummy_idx = torch.arange(self.num_nodes)
        dummy_idx = dummy_idx.unsqueeze(0).expand(self.batch_size, self.num_nodes)
        dummy_static_features = torch.zeros((self.batch_size, self.num_nodes, self.num_static_features), dtype=torch.float32)
        static_features = torch.zeros((self.batch_size, self.num_items, self.num_static_features), dtype=torch.float32)

        origin_coords = self.norm_coords[:, 0, :].unsqueeze(1)
        item_coords = self.norm_coords[self.item_batch_idx.ravel(), self.item_city_idx.ravel(), :].view(self.batch_size, self.num_items, 2)
        item_dist_to_origin = torch.norm(origin_coords-item_coords, dim=2)
        static_features[:, :, 0] = item_dist_to_origin
        static_features[:, :, 1] = self.norm_weights
        static_features[:, :, 2] = self.norm_profits
        static_features[:, :, 3] = self.norm_profits/self.norm_weights
        
        dummy_dist_to_origin = torch.norm(origin_coords-self.norm_coords, dim=2)
        dummy_static_features[:, :, 0] = dummy_dist_to_origin

        static_features = torch.cat((static_features, dummy_static_features), dim=1)
        return static_features

        # dist_to_curr, current_weight, current_velocity
    def get_dynamic_features(self):
        self.num_dynamic_features = 3
        # per item features = distance
        current_coords = self.norm_coords[self.batch_idx, self.current_location, :].unsqueeze(1)
        item_coords = self.norm_coords[self.item_batch_idx.ravel(), self.item_city_idx.ravel(), :].view(self.batch_size, self.num_items, 2)
        item_dist_to_curr = torch.norm(current_coords-item_coords, dim=2)
        dummy_dist_to_curr = torch.norm(self.norm_coords-current_coords, dim=2)
        dist_to_curr = torch.cat((item_dist_to_curr, dummy_dist_to_curr), dim=1)
        dist_to_curr = dist_to_curr.unsqueeze(2)

        # global features weigh and velocity
        global_dynamic_features = torch.zeros((self.batch_size, 2))
        global_dynamic_features[:, 0] = torch.sum(self.norm_weights*self.item_selection, dim=1)
        current_vel = self.max_v - (self.current_load/self.max_cap)*(self.max_v-self.min_v)
        current_vel = torch.maximum(current_vel, self.min_v)
        global_dynamic_features[:, 1] = current_vel
        global_dynamic_features = global_dynamic_features.unsqueeze(1)
        global_dynamic_features = global_dynamic_features.repeat_interleave(self.num_items+self.num_nodes,dim=1)
        dynamic_features = torch.cat([dist_to_curr, global_dynamic_features], dim=2)
        return dynamic_features

    def act(self, active_idx, selected_idx):
        # filter which is taking item, which is visiting nodes only
        active_idx = active_idx.cpu()
        selected_idx = selected_idx.cpu()
        is_taking_item = selected_idx < self.num_items
        if torch.any(is_taking_item):
            self.take_item(active_idx[is_taking_item], selected_idx[is_taking_item])

        is_visiting_node_only = torch.logical_not(is_taking_item)
        if torch.any(is_visiting_node_only):
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
        items_more_than_cap = torch.zeros_like(self.eligibility_mask, dtype=torch.bool)
        items_more_than_cap[:, :self.num_items] = self.weights > current_cap.unsqueeze(1)
        self.eligibility_mask = torch.logical_and(self.eligibility_mask, ~items_more_than_cap)
            
        # check if the selected item's location is not the current location too
        selected_item_location = self.item_city_idx[active_idx, selected_item]
        is_diff_location = self.current_location[active_idx] != selected_item_location
        if torch.any(is_diff_location):
            self.visit_node(active_idx[is_diff_location], selected_item_location[is_diff_location])

    def visit_node(self, active_idx, selected_node):
        # set is selected
        self.is_selected[active_idx, selected_node+self.num_items] = True
        # set dummy item for the selected location is infeasible
        self.eligibility_mask[active_idx, selected_node+self.num_items] = False
        
        # mask all current locations' item as ineligible before moving
        
        current_locations_item_mask = self.all_city_idx[active_idx, :] == self.current_location[active_idx].unsqueeze(1)
        self.eligibility_mask[active_idx,:] = torch.logical_and(self.eligibility_mask[active_idx,:], ~current_locations_item_mask)

        # set current location to next location
        self.current_location[active_idx] = selected_node

        # save it to tour list
        self.tour_list[active_idx, self.num_visited_nodes[active_idx]] = selected_node
        self.num_visited_nodes[active_idx] += 1

        #check if all nodes are visited, if yes then make the dummy item for first city feasibe
        is_all_visited = self.num_visited_nodes == self.num_nodes+1
        if torch.any(is_all_visited):
            self.eligibility_mask[is_all_visited, self.num_items] = True


    def finish(self):
        # computing tour lenghts or travel time
        tour_A = self.tour_list
        tour_B = tour_A.roll(-1)
        batch_idx_ = self.batch_idx.unsqueeze(1).expand_as(self.tour_list)
        edge_lengths = self.W[batch_idx_, tour_A, tour_B]
        selected_weights = self.item_selection.float()*self.weights
        selected_node_weights = torch.zeros((self.batch_size, self.num_nodes), dtype=torch.float32)
        selected_node_weights = selected_node_weights.scatter_add(1, self.item_city_idx, selected_weights)
        ordered_selected_weights = selected_node_weights[self.node_batch_idx.view_as(self.tour_list), self.tour_list]
        final_weights = torch.cumsum(ordered_selected_weights, dim=1)
        velocity_list = self.max_v.unsqueeze(1) - (final_weights/self.max_cap.unsqueeze(1))*(self.max_v-self.min_v).unsqueeze(1)
        # velocity_list[velocity_list<self.min_v.unsqueeze(1)] = self.min_v.unsqueeze(1)
        tour_lengths = (edge_lengths/velocity_list).sum(dim=-1)

        # total profit
        selected_profits = self.item_selection*self.profits
        total_profits = selected_profits.sum(dim=-1)

        total_cost = total_profits - tour_lengths*self.renting_rate
        return self.tour_list, self.item_selection, tour_lengths, total_profits, total_cost