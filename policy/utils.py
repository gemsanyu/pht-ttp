import functools
from random import random
import torch
import math
import inspect

from typing import NamedTuple

CPU_DEVICE = torch.device('cpu')


def get_utility(n=10):
    n = float(n)
    idx = torch.arange(n, dtype=torch.float32) + 1
    u = math.log(n/2 + 1) - torch.log(idx)
    u = u * (1-(u < 0).float())
    u = u / torch.sum(u)
    u = u - 1/n
    return u


def get_utility_hansen(n=10):
    a = math.log(n/2. + 1.)
    rank = torch.arange(n) + 1
    b = torch.log(n-rank)
    utility = torch.clamp_min(a-b, 0)
    return utility


# solution list must contain only non-dominated solutions.
# make sure to use the same random points on computing HV contributions,
# so that the HV becomes consistent
def get_hypervolume_mc(solution_list, random_points, reference_point=None, device=CPU_DEVICE):
    N, M = solution_list.shape
    if reference_point is None:
        reference_point = torch.ones(
            (1, M), dtype=torch.float32, device=device) * 1.1
        # reference_point = torch.ones((1,M), dtype=torch.float32, device=device) + 1./max(N-1.,1.)
    Nr, _ = random_points.shape
    R = random_points
    # R = torch.rand((Nr, M), dtype=torch.float32, device=device)
    R = R*reference_point

    idx_r = torch.arange(Nr)
    cmp_mat = torch.repeat_interleave(idx_r.unsqueeze(0), N, dim=0)

    idx_n = torch.arange(N)
    cmp_matn = torch.repeat_interleave(idx_n.unsqueeze(1), Nr, dim=1)

    # randomized points geq in obj 1
    geq1 = R[cmp_mat, 0] >= solution_list[cmp_matn, 0]
    # randomized points geq in obj 2
    geq2 = R[cmp_mat, 1] >= solution_list[cmp_matn, 1]
    # randomized points greater in obj 1
    gt1 = R[cmp_mat, 0] > solution_list[cmp_matn, 0]
    # randomized points greater in obj 2
    gt2 = R[cmp_mat, 1] > solution_list[cmp_matn, 1]

    # greater equal in both, gt in one of obj
    geq = torch.logical_and(geq1, geq2)
    gt = torch.logical_or(gt1, gt2)
    is_dominated = torch.logical_and(geq, gt)
    is_dominated = torch.any(is_dominated, dim=0)
    dominated_count = torch.sum(is_dominated).float()
    # is_dominated = torch.any(is_dominated, dim=1)
    hv_mc = dominated_count/Nr
    return hv_mc

# exact, O(n) for 2d (only for 2d ya)
def get_hypervolume(solution_list, reference_point=None, device=CPU_DEVICE):
    N, M = solution_list.shape
    if reference_point is None:
        reference_point = torch.ones(
            (1, M), dtype=torch.float32, device=device) * 1.1

    # sort solution list
    val, sorted_idx = torch.sort(solution_list[:, 0], dim=0)
    sorted_solution_list = solution_list[sorted_idx, :]
    hv = 0.
    for i in range(len(sorted_solution_list)):
        hv = hv + (reference_point[0, 0]-sorted_solution_list[i, 0]) * \
            (reference_point[0, 1]-sorted_solution_list[i, 1])
        reference_point[0, 1] = sorted_solution_list[i, 1]

    return hv


def get_hv_contributions(solution_list, num_random_points=1000000, reference_point=None, device=CPU_DEVICE):
    num_solutions, M = solution_list.shape
    # random_points = torch.rand(
    #     (num_random_points, M), dtype=torch.float32, device=device) * 1.1
    if reference_point is not None:
        solution_list /= reference_point
    # total_hv = get_hypervolume_mc(
    #     solution_list, random_points, reference_point=None, device=device)
    total_hv = get_hypervolume(solution_list)
    if num_solutions == 1:
        return total_hv
    hv_contributions = torch.zeros(
        (num_solutions,), dtype=torch.float32, device=device)
    solution_mask = torch.ones(
        (num_solutions,), dtype=torch.bool, device=device)
    for i in range(num_solutions):
        solution_mask[i] = 0
        # hv_without_sol = get_hypervolume_mc(
        #     solution_list[solution_mask, :], random_points, reference_point=None, device=device)
        hv_without_sol = get_hypervolume(solution_list[solution_mask, :])
        hv_contributions[i] = total_hv-hv_without_sol
        solution_mask[i] = 1
    return hv_contributions


def nondominated_sort(solution_list):
    N, M = solution_list.shape
    idx = torch.arange(N)
    idx1 = torch.repeat_interleave(idx, N-1)
    idx2 = torch.repeat_interleave(idx.unsqueeze(0), N, dim=0)
    idx2 = idx2.masked_select(~torch.eye(N, dtype=torch.bool))
    # compare first obj geq
    geq1 = solution_list[idx1, 0] >= solution_list[idx2, 0]
    # #compare second obj geq
    geq2 = solution_list[idx1, 1] >= solution_list[idx2, 1]
    # # compare first obj greater
    gt1 = solution_list[idx1, 0] > solution_list[idx2, 0]
    # #compare second obj greater
    gt2 = solution_list[idx1, 1] > solution_list[idx2, 1]

    # greater equal in both, gt in one of obj
    geq = torch.logical_and(geq1, geq2)
    gt = torch.logical_or(gt1, gt2)
    is_dominated = torch.logical_and(geq, gt).view(N, N-1)
    is_dominated = torch.any(is_dominated, dim=1)
#   P_nondom_idx = torch.arange(N)[~is_dominated]
    return ~is_dominated


def update_nondom_archive(curr_nondom_archive, f_list):
    if curr_nondom_archive is not None:
        f_list = torch.cat([curr_nondom_archive, f_list])
    is_nondom = nondominated_sort(f_list)
    return f_list[is_nondom]

def combine_with_nondom(f_list, nondom_archive):
    exists_in_flist = torch.eq(nondom_archive.unsqueeze(1), f_list)
    exists_in_flist = exists_in_flist.any(dim=2).any(dim=1)
    combined_unique = torch.cat([f_list, nondom_archive[torch.logical_not(nondom_archive)]])
    return combined_unique

class BatchProperty(NamedTuple):
    num_nodes: int
    num_items_per_city: int
    num_clusters: int
    item_correlation: int
    capacity_factor: int


def get_batch_properties(num_nodes_list, num_items_per_city_list):
    """
        training dataset information for each batch
        1 batch will represent 1 possible problem configuration
        including num of node clusters, capacity factor, item correlation
        num_nodes, num_items_per_city_list
    """
    batch_properties = []
    capacity_factor_list = [i+1 for i in range(10)]
    num_clusters_list = [1]
    item_correlation_list = [i for i in range(3)]

    for num_nodes in num_nodes_list:
        for num_items_per_city in num_items_per_city_list:
            for capacity_factor in capacity_factor_list:
                for num_clusters in num_clusters_list:
                    for item_correlation in item_correlation_list:
                        batch_property = BatchProperty(num_nodes, num_items_per_city,
                                                       num_clusters, item_correlation,
                                                       capacity_factor)
                        batch_properties += [batch_property]
    return batch_properties


def custom_permute_2d(x, permutation):
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret


def print_gpu_memory_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    caller_name = inspect.stack()[1][3]
    print("------------------------")
    print("In Function:", caller_name)
    print("CUDA MEMORY")
    print("TOTAL MEMORY:", t)
    print("RESERVED:", r)
    print("FREE in RESERVER", f)
    print("------------------------")


# NOVELTY MEASURE (BC MEASURE)
# SNN simcos distance metric
# primary distance metric is L2-Norm
def simcos(A, k=50):
    num_sample, d = A.shape
    k = min(k, num_sample)
    dist_matrix = torch.cdist(A, A)
    nearest_dists, NN = torch.topk(dist_matrix, k=k, dim=1, largest=False)
    NN_one_hot = torch.nn.functional.one_hot(NN, num_classes=num_sample)
    NN_one_hot = torch.sum(NN_one_hot, dim=1).long()
    SNN_matrix = torch.logical_and(NN_one_hot.unsqueeze(1), NN_one_hot).long()
    SNN = torch.sum(SNN_matrix, dim=2)
    return SNN/k


def bc_item_selection(item_selection_list):
    return item_selection_list.float()


def bc_node_order(node_order_list):
    '''
    give max_score (which is num_nodes), to the first visited,
    then give num_nodes-1 score to second visited, etc...
    ofc first visited is always 0

    then  divide by num_nodes -> we get probability of visitation, which 
    is somewhat similar to what is used in rl algo for tsp-like problems
    '''
    num_sample, num_nodes = node_order_list.shape
    score_list = torch.arange(num_nodes, dtype=torch.float32) + 1
    score_list = score_list.flip(dims=(0,))

    total_score = torch.sum(score_list)
    bc = score_list[node_order_list]
    bc /= num_nodes
    return bc


def get_bc_var(node_order_list, item_selection_list):
    bc_node = bc_node_order(node_order_list)
    mean_bc_node = torch.mean(bc_node, dim=0, keepdim=True)
    var_bc_node = (bc_node-mean_bc_node)**2
    bc_node_final = torch.mean(var_bc_node, dim=1, keepdim=True)
    bc_item = bc_item_selection(item_selection_list)
    mean_bc_item = torch.mean(bc_item, dim=0, keepdim=True)
    var_bc_item = (bc_item-mean_bc_item)**2
    bc_item_final = torch.mean(var_bc_item, dim=1, keepdim=True)
    bc = 1-(bc_node_final+bc_item_final)/2
    return bc


# nsga2
def get_domination_matrix(score_list):
    is_smaller = score_list.unsqueeze(1) < score_list
    is_smaller_eq = score_list.unsqueeze(1) <= score_list
    is_smaller = torch.any(is_smaller, dim=2)
    is_smaller_eq = torch.all(is_smaller_eq, dim=2)
    is_dominate = torch.logical_and(is_smaller, is_smaller_eq)
    return is_dominate


def get_domination_fronts(score_list):
    num_sample, num_obj = score_list.shape
    index_list = torch.arange(num_sample)
    domination_matrix = get_domination_matrix(score_list)
    fronts = []
    while len(index_list) > 0:
        is_dominated = torch.sum(domination_matrix, dim=0) > 0
        is_non_dominated = torch.logical_not(is_dominated)
        fronts = fronts + [index_list[is_non_dominated]]
        # remove nondominated solutions from dom matrix
        index_list = index_list[is_dominated]
        domination_matrix = domination_matrix[is_dominated, :][:, is_dominated]

    return fronts


def get_crowding_distance(score_list, fmax, fmin):
    num_sample, num_obj = score_list.shape
    if num_sample <= 2:
        return torch.zeros((num_sample,), dtype=torch.float32) + 999999
    arg_idx = torch.argsort(score_list, dim=0)
    obj_rank = torch.argsort(arg_idx, dim=0)  # scipy rankdata
    val_range = (fmax-fmin).view(1, num_obj)
    sorted_score_list, _ = torch.sort(score_list, dim=0)
    next_obj = sorted_score_list.gather(0, (obj_rank+1) % num_sample)
    prev_obj = sorted_score_list.gather(0, (obj_rank-1) % num_sample)
    dist_obj = next_obj-prev_obj
    is_extreme = torch.logical_or(obj_rank == 0, obj_rank == (num_sample-1))
    is_extreme = torch.sum(is_extreme, dim=1).bool()
    dist_obj /= val_range
    dist_obj = torch.sum(dist_obj, dim=1)
    dist_obj[is_extreme] = 999999
    return dist_obj


def cmp(a, b, front_idx, crowding_distance_list):
    if front_idx[a] < front_idx[b]:
        return -1
    elif front_idx[a] > front_idx[b]:
        return 1
    if crowding_distance_list[a] > crowding_distance_list[b]:
        return -1
    elif crowding_distance_list[a] < crowding_distance_list[b]:
        return 1
    return 0


# nsga2
def get_nondominated_rank(score_list):
    real_num_sample = len(score_list)
    # if nondom_archive is not None:
    #     # print(score_list, nondom_archive)
    #     score_list = torch.cat([score_list, nondom_archive], dim=0)
    num_sample, num_obj = score_list.shape
    fronts = get_domination_fronts(score_list)
    front_idx = torch.zeros((num_sample,))
    for i in range(len(fronts)):
        front_idx[fronts[i]] = i
    fmax, max_idx = torch.max(score_list, dim=0)
    fmin, min_idx = torch.min(score_list, dim=0)
    crowding_distance_list = torch.zeros((num_sample,), dtype=torch.float32)
    for front in fronts:
        crowding_distance_list[front] = get_crowding_distance(
            score_list[front], fmax, fmin)

    cmp_new = functools.partial(
        cmp, front_idx=front_idx, crowding_distance_list=crowding_distance_list)
    cmp_key = functools.cmp_to_key(cmp_new)
    idx_list = list(range(num_sample))
    # idx_list.sort(key=cmp_key)
    idx_list.sort(key=lambda i: (front_idx[i], -crowding_distance_list[i]))
    idx_list = torch.Tensor(idx_list)
    # rank by argsort one more time
    idx_list = torch.argsort(idx_list)
    idx_list = idx_list[:real_num_sample]
    return idx_list


def get_score_hv_contributions(f_list, negative_hv, nondom_archive=None, reference_point=None):
    real_num_sample, M = f_list.shape
    if nondom_archive is not None:
        f_list = combine_with_nondom(f_list, nondom_archive)
    num_sample, _ = f_list.shape
    
    # count hypervolume, first nondom sort then count, assign penalty hv too
    hv_contributions = torch.zeros(
        (num_sample,), dtype=torch.float32)
    is_nondom = nondominated_sort(f_list)
    hv_contributions[is_nondom] = get_hv_contributions(f_list[is_nondom, :], reference_point=reference_point)
    hv_contributions[~is_nondom] = negative_hv
    # hv_contributions = (1-novelty_w)+hv_contributions + novelty_w*novelty_score

    # prepare utility score
    score = hv_contributions.unsqueeze(1)
    score = score[:real_num_sample]
    noted = torch.cat([score, 1.1-f_list/reference_point], dim=-1)
    print(noted)
    return score

def get_score_nsga2(f_list, nondom_archive=None, reference_point=None):
    real_num_sample, M = f_list.shape
    if nondom_archive is not None:
        f_list = combine_with_nondom(f_list, nondom_archive)
    num_sample, _ = f_list.shape
    utility = get_utility(num_sample)
    rank = get_nondominated_rank(f_list)
    score = utility[rank][:real_num_sample].unsqueeze(1)
    return score