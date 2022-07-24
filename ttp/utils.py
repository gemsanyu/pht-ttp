import torch
from ortools.algorithms import pywrapknapsack_solver
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

KNAPSACK_SOLVER = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.
    KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
    # KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 
    'KnapsackExample')

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


# get renting rate by solving both knapsack and TSP
def get_renting_rate(W, weights, profits, capacity):
    # solve the knapsack first
    optimal_profit = solve_knapsack(weights, profits, capacity)
    # solve the tsp
    optimal_tour_length = solve_tsp(W)
    renting_rate = float(optimal_profit)/float(optimal_tour_length)
    return optimal_profit, optimal_tour_length, renting_rate


def solve_knapsack(weights, profits, capacity):
    weights_list = [weights.long().tolist()]
    profits_list = profits.long().tolist()
    cap = [capacity.long().item()]
    solver = KNAPSACK_SOLVER
    
    solver.Init(profits_list, weights_list, cap)
    optimal_profit = solver.Solve()
    return optimal_profit


def solve_tsp(W):
    data = {}
    data["distance_matrix"] = W.tolist()
    data["num_vehicles"] = 1
    data["depot"] = 0
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
     # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    return solution.ObjectiveValue()