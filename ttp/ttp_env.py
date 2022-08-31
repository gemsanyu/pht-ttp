from platform import node
from typing import Optional, Tuple, NamedTuple
import torch
import numpy as np


class ItemState(NamedTuple):
    active_idx: torch.Tensor
    raw_static_features: torch.Tensor
    raw_dynamic_features: torch.Tensor
    eligibility_mask: torch.Tensor
    prev_selected_idx: torch.Tensor    

class NodeState(NamedTuple):
    active_idx: torch.Tensor
    raw_static_features: torch.Tensor
    raw_dynamic_features: torch.Tensor
    eligibility_mask: torch.Tensor
    prev_selected_idx: torch.Tensor

class State(NamedTuple):
    item_state: Optional[ItemState]=None
    node_state: Optional[NodeState]=None

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
        self.num_items_per_city = int(np.sum(self.item_city_mask[0,1]))
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
        self.is_selecting_item = np.zeros((self.batch_size,), dtype=bool)
        self.raw_item_static_features = self.get_item_static_features()
        self.raw_node_static_features = self.get_node_static_features()        
        # idx == -1, then first selection, use learned initial embeddings    
        self.prev_selected_node_idx = np.zeros((self.batch_size,), dtype=np.int64) - 1
        self.prev_selected_item_idx = np.zeros((self.batch_size,), dtype=np.int64) - 1

        # weight, profit, density
    def get_item_static_features(self):
        item_static_features = np.zeros((self.batch_size, self.num_items, 3), dtype=np.float32)
        item_static_features[:, :, 0] = self.norm_weights
        item_static_features[:, :, 1] = self.norm_profits
        item_static_features[:, :, 2] = self.norm_profits/self.norm_weights
        return item_static_features

        # distance to origin, sum of items' density in that node
    def get_node_static_features(self):
        node_static_features = np.zeros((self.batch_size, self.num_nodes, 2), dtype=np.float32)
        origin_coords = np.expand_dims(self.norm_coords[:,0,:],axis=1)
        dist_to_origin = np.linalg.norm(self.coords-origin_coords, axis=2)
        node_static_features[:, :, 0] = dist_to_origin
        density = self.norm_profits/self.norm_weights
        node_density = self.item_city_mask * density[:, np.newaxis, :]
        node_density = np.sum(node_density, axis=-1)
        node_static_features[:, :, 1] = node_density
        return node_static_features

    # current_cap, current_vel
    def get_item_dynamic_features(self):
        item_dynamic_features = np.zeros((self.batch_size, 2), dtype=np.float32)
        current_cap = self.current_load/self.max_cap
        item_dynamic_features[:, 0] = current_cap
        current_vel = self.max_v - (current_cap)*(self.max_v-self.min_v)
        current_vel = np.maximum(current_vel, self.min_v)  
        item_dynamic_features[:, 1] = current_vel
        # item_dynamic_features[:, 2] = np.sum(self.norm_profits*self.item_selection, axis=1)
        item_dynamic_features = np.repeat(item_dynamic_features[:, np.newaxis, :], self.num_items_per_city, 1)
        return item_dynamic_features

    # travel time to current, travel time to origin, current_cap, current_vel
    def get_node_dynamic_features(self):
        node_dynamic_features = np.zeros((self.batch_size, self.num_nodes, 4), dtype=np.float32)
        origin_coords = np.expand_dims(self.norm_coords[:,0,:],axis=1)
        current_coords = np.take_along_axis(self.norm_coords, self.current_location[:,np.newaxis,np.newaxis], 1)
        current_cap = self.current_load/self.max_cap
        current_vel = self.max_v - (current_cap)*(self.max_v-self.min_v)
        current_vel = np.maximum(current_vel, self.min_v)  
        # per node features
        trav_time_to_current = np.linalg.norm(current_coords-self.norm_coords, axis=2)/current_vel[:, np.newaxis]
        trav_time_to_origin = np.linalg.norm(origin_coords-self.norm_coords, axis=2)/current_vel[:, np.newaxis]
        node_dynamic_features[:,:,0] = trav_time_to_current
        node_dynamic_features[:,:,1] = trav_time_to_origin
        # global features
        # current_profits = np.sum(self.norm_profits*self.item_selection, axis=1)
        node_dynamic_features[:,:,2] = current_cap[:, np.newaxis]
        node_dynamic_features[:,:,3] = current_vel[:, np.newaxis]
        # node_dynamic_features[:,:,4] = current_profits[:, np.newaxis]
        return node_dynamic_features
    
    def get_current_state(self):
        is_active = np.sum(self.eligibility_mask, axis=1, dtype=bool)
        is_env_finished = np.any(is_active) is False
        item_state = None
        node_state = None
        if is_env_finished:
            return State(None, None)

        active_taking_item = np.logical_and(is_active, self.is_selecting_item)
        active_taking_item_idx = np.nonzero(active_taking_item)[0]
        if len(active_taking_item_idx) > 0:
            # index only the active idx
            # get only the current location's items' features and masking
            raw_item_static_features = self.raw_item_static_features[active_taking_item_idx]
            raw_item_dynamic_features = self.get_item_dynamic_features()[active_taking_item_idx]
            active_current_location = self.current_location[active_taking_item_idx]
            active_current_item_city_mask = self.item_city_mask[active_taking_item_idx, active_current_location]
            item_eligibility_mask = self.eligibility_mask[:, :self.num_items][active_taking_item_idx]
            item_eligibility_mask = item_eligibility_mask[active_current_item_city_mask].reshape((len(active_taking_item_idx), self.num_items_per_city))
            prev_selected_item_idx = self.prev_selected_item_idx[active_taking_item_idx]
            item_state = ItemState(active_taking_item_idx, raw_item_static_features, raw_item_dynamic_features, item_eligibility_mask, prev_selected_item_idx)

        active_visiting_node = np.logical_and(is_active, np.logical_not(self.is_selecting_item))
        active_visiting_node_idx = np.nonzero(active_visiting_node)[0]
        if len(active_visiting_node_idx)>0:
            raw_node_static_features = self.raw_node_static_features[active_visiting_node_idx]
            raw_node_dynamic_features = self.get_node_dynamic_features()[active_visiting_node_idx]
            node_eligibility_mask = self.eligibility_mask[:, self.num_items:][active_visiting_node_idx]
            prev_selected_node_idx = self.prev_selected_node_idx[active_visiting_node_idx]
            node_state = NodeState(active_visiting_node_idx, raw_node_static_features, raw_node_dynamic_features, node_eligibility_mask, prev_selected_node_idx)

        return State(item_state, node_state)

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

        # dynamic_features = self.get_dynamic_features()
        # return dynamic_features, self.eligibility_mask


    """
        1. filter the taken items  
        if dummy item is taken, then just set the batch flag
        to is_taking_item = False
        2. also convert the item index to original index
        from [0, num_city_per_idx-1] to [0, num_items-1]
    """
    def take_item_with_dummy(self, active_idx, selected_item_with_dummy_idx):
        selected_item_with_dummy_idx = selected_item_with_dummy_idx.cpu().numpy()
        is_selecting_dummy = selected_item_with_dummy_idx == self.num_items_per_city
        is_not_selecting_dummy = np.logical_not(is_selecting_dummy)
        not_dummy_current_location = self.current_location[active_idx][is_not_selecting_dummy]
        selected_item_idx = selected_item_with_dummy_idx[is_not_selecting_dummy]
        real_selected_item_idx = not_dummy_current_location+selected_item_idx*(self.num_nodes-1)-1
        active_idx_selecting_dummy = active_idx[is_selecting_dummy]
        active_idx_not_selecting_dummy = active_idx[is_not_selecting_dummy]
        # if select dummy then set flag to not selecting item anymore
        # and mask all current locations' item to false
        if len(active_idx_selecting_dummy) > 0:
            self.is_selecting_item[active_idx_selecting_dummy] = False
            # mask all current locations' item as ineligible before moving
            current_locations_item_mask = self.all_city_idx[active_idx_selecting_dummy, :] == self.current_location[active_idx_selecting_dummy][:, np.newaxis]
            self.eligibility_mask[active_idx_selecting_dummy,:] = np.logical_and(self.eligibility_mask[active_idx_selecting_dummy,:], ~current_locations_item_mask)
        
        #select items
        if len(active_idx_not_selecting_dummy)>0:
            self.take_item(active_idx_not_selecting_dummy, real_selected_item_idx)

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
        selected_node = selected_node.cpu().numpy()
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

        # after visiting node, must be selecting items
        self.is_selecting_item[active_idx] = True

        self.prev_selected_node_idx[active_idx] = selected_node

        #check if all nodes are visited, if yes then make the dummy item for first city feasibe
        # not needed in spn, keep the origin nod unvisitable
        # is_all_visited = self.num_visited_nodes == self.num_nodes+1
        # if np.any(is_all_visited):
        #     self.eligibility_mask[is_all_visited, self.num_items] = True

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