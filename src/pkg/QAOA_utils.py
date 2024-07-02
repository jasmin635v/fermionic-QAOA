from pennylane import numpy as np
import cmath
import statistics
import networkx as nx
from itertools import combinations
from collections import Counter
import math

matchgateOnes = np.array([[1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1]])

fSwap = np.array([[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1]])

fRyy = 1 / np.sqrt(2) * np.array([[1, 0, 0, -1],
                    [0, 1,  -1, 0],
                    [0, 1, 1, 0],
                    [1, 0, 0, 1]])


def Rzz_matrice(gamma):
    exp_term = cmath.exp(1j * gamma/ 2)
    exp_term_m = cmath.exp(-1j * gamma/ 2)
    return np.array([[exp_term_m, 0, 0, 0],
            [0, exp_term, 0, 0],
            [0, 0, exp_term, 0],
            [0, 0, 0, exp_term_m]])

def generate_string_graph_representation(graph):
    # Initialize an empty list to store the result strings
    result_strings = []

    # Sort the graph based on the first element of each tuple (node)
    graph_sorted = sorted(graph)

    # Initialize variables to keep track of current node and components
    current_node = graph_sorted[0][0]
    component = [current_node]

    for edge in graph_sorted:
        if edge[0] == current_node + 1:
            component.append(edge[1])
        else:
            result_strings.append('-'.join(map(str, component)))
            current_node = edge[0]
            component = [current_node, edge[1]]

    # Append the last component to result_strings
    result_strings.append('-'.join(map(str, component)))

    return result_strings
# unitary operator U_B with parameter beta

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)


def bitstring_to_objective(bitstring, graph):
    #convert bitstring to a list of 0 and 1
    binary_list = [int(char) for char in bitstring]
    obj = 0
    for edge in graph:
        # objective for the MaxCut problem
        obj += 1 - (binary_list[edge[0]] * binary_list[edge[1]])
    return obj * 0.5

def int_to_bitstring(num):
    # Convert integer to binary string (without '0b' prefix)
    bit_string = bin(num)[2:]
    return bit_string


def compute_stats(numbers):

    weighted_sum = sum([number[0]*number[1] for number in numbers])
    total_count = sum([number[1] for number in numbers])
    mean = weighted_sum/total_count
    
    minimum = min([number[0] for number in numbers])
    maximum = max([number[0] for number in numbers])

    # Step 2: Calculate weighted variance
    weighted_variance_sum = sum(number[1]  * (number[0]  - mean)**2 for number in numbers)
    weighted_variance = weighted_variance_sum / total_count

    # Step 3: Calculate weighted standard deviation
    weighted_stddev = math.sqrt(weighted_variance)

    sorted_numbers = sorted(numbers, key=lambda x: x[1], reverse=True)
    most_common_element = sorted_numbers[0][0]
    most_common_count = sorted_numbers[0][1]

    most_common_element_count_ratio = most_common_count / total_count
    #weighted_mean_3 = (sorted_numbers[0][0]*sorted_numbers[0][1] + sorted_numbers[1][0]*sorted_numbers[1][1] +sorted_numbers[2][0]*sorted_numbers[2][1])  / (sorted_numbers[0][1] + sorted_numbers[1][1] +sorted_numbers[2][1])

    return  most_common_element, most_common_element_count_ratio, mean, maximum, weighted_stddev


def generate_all_graphs(n):
    """ Generate all possible graphs with n vertices """
    all_graphs = []
    nodes = range(n)
    
    # Iterate over all possible combinations of edges
    for num_edges in range(1,n*(n-1)//2 + 1):  # Maximum number of edges for n vertices 

        all_graphs_with_num_edges = []

        possible_edges = list(combinations(nodes, 2))
        
        # Generate all combinations of edges for the current num_edges
        edges_combinations = [[edge for edge in combo] for combo in list(combinations(possible_edges, num_edges))]

        # Filter out combinations that do not include all nodes
        combinations_with_all_nodes = [
            combo for combo in edges_combinations 
            if set(nodes) <= set(node for edge in combo for node in edge)
        ]

        if not combinations_with_all_nodes:  # Check if the list is empty
            continue  # Skip to the next iteration if the list is empty

        
        # Add graph from combination to list
        
        for combination in combinations_with_all_nodes:
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(combination)
            graph_absent = True
            # check if an isomorphic graph is already present. if so, increment the weight of that graph. If not, add first graph
            for present_graph in all_graphs_with_num_edges:
                if nx.is_isomorphic(present_graph[0],G):
                    present_graph[1] += 1
                    graph_absent = False
                    break
            
            if graph_absent: #if no isomorphic graph found, add it with weight 1
                all_graphs_with_num_edges.append([G,1])

        edges_combination_weight = [(list(graph[0].edges()),graph[1]) for graph in all_graphs_with_num_edges]
            
        all_graphs.extend(edges_combination_weight)

    return all_graphs
