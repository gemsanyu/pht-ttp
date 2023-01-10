import torch
from ortools.algorithms import pywrapknapsack_solver
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

KNAPSACK_SOLVER = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.
    KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
    # KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 
    'KnapsackExample')

def solve_knapsack(weights, profits, capacity):
    weights_list = [weights.long().tolist()]
    profits_list = profits.long().tolist()
    cap = [capacity.long().item()]
    solver = KNAPSACK_SOLVER
    
    solver.Init(profits_list, weights_list, cap)
    optimal_profit = solver.Solve()
    sol_idx = []
    for i in range(len(weights)):
        if solver.BestSolutionContains(i):
            sol_idx += [i]
    item_selection = torch.zeros((len(weights),), dtype=torch.bool)
    item_selection[sol_idx] = True
    return optimal_profit, item_selection


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
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)
    search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH)
    search_parameters.time_limit.seconds = 3
     # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    route_list = print_solution(manager, routing, solution)
    return route_list, solution.ObjectiveValue()

def print_solution(manager, routing, solution):
    """Prints solution on console."""
    # print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    route_list =[]
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        route_list += [manager.IndexToNode(index)]
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    plan_output += 'Route distance: {}miles\n'.format(route_distance)
    # print(plan_output)
    return torch.tensor(route_list)
    

def solve_tsp_memoization(W):
    import numpy as np
    memo = [-1 for _ in range(2**(len(W)))]
    print(len(memo))
    is_visited = [False for _ in range(2**(len(W)))]
    tmp_2 = np.asanyarray([2**i for i in range(len(W))])
    def solve_(cur_idx, mask:np.array):
        bin_idx = int((tmp_2*mask).sum())
        if is_visited[bin_idx]:
            return memo[bin_idx]
        if np.all(mask):
            return W[cur_idx][0]
        min_length = 9999999999999
        for i in range(len(W)):
            if mask[i] == 1:
                continue
            mask_ = np.copy(mask)
            mask_[i] = 1
            route_length = solve_(i, mask_) + W[cur_idx, i]
            if min_length > route_length:
                min_length = route_length
        memo[bin_idx] = min_length
        is_visited[bin_idx]= True
        return min_length

    mask = np.asanyarray([0 for _ in range(len(W))])
    mask[0] = 1
    best_route_length = solve_(0, mask)
    return best_route_length

if __name__ == "__main__":
    num_nodes = 6
    coords = torch.randint(0,10, size=(num_nodes,2), dtype=torch.float32)
    W = torch.cdist(coords, coords, p=2)
    best_route_length = solve_tsp_memoization(W)
    print(W)
    print(best_route_length)
    route_list, _ = solve_tsp(W*10000)
    # print(float(best_route_length2)/10000.)
    route_A = route_list
    route_B = route_list.roll(-1)
    print(route_A, route_B)
    best_route_length2 = W[route_A, route_B].sum()
    print(best_route_length2)
