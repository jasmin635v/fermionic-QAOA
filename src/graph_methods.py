import networkx as nx
from itertools import combinations, groupby
import math, time

# connected (no free vertices)  graphs with n nodes:

#Nodes     2   3    4      5        6          7            8              9
#Unlabeled 1,  2,   6,    21,     112,       853,       11117,         261080
#Labeled   4, 38, 728, 26704, 1866256, 251548592, 66296291072, 34496488594816

#connected means that no vertice has no edge, not that all vertices have a path between then

def plot_xylists(xy_lists):
    num_subplots = len(xy_lists)
    if num_subplots == 0:
        return
    
    #fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 16))

    # Iterate over list_of_list_of_data and plot each list_of_data as a subplot
    for i, list_of_data in enumerate(xy_lists):
        title = list_of_data[0][2]
        x_values = [pair[0] for pair in list_of_data]
        y_values = [pair[1] for pair in list_of_data]
        # axes[i].scatter(x_values, y_values, color='blue', marker='o', s=50, alpha=0.8)
        # axes[i].set_title(f'Graph with {title} Vertices')
        # axes[i].set_xlabel('Number of edges')
        # axes[i].set_ylabel('Mean automorphism number')
        # axes[i].set_xlim(left=0)     
        # axes[i].grid(True)


    # Adjust layout and display the plot
    # fig.subplots_adjust(hspace=1)
    # plt.tight_layout()
    # plt.show()

def return_graph_from_combination(combination, nodes = None):

    if nodes == None:
        max_node_indice = node_number_from_graph(combination)
        nodes = list(range(0,max_node_indice +1))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(combination)
    
    return G

def node_number_from_graph(combination):
        node_list = list(set(node for edge in combination for node in edge))
        max_node_indice = max(node_list)
        return max_node_indice


def generate_all_n1_graphs_from_n_graph(graph): # graph format [(0, 1), (2, 3)]

    max_vertice = max(max(edge) for edge in graph)
    new_vertice = max_vertice + 1

    #get all possible tuples
    possible_tuples = [(node,new_vertice) for node in range(0,new_vertice)]

    all_combinations = []
    for r in range(1, len(possible_tuples) + 1):
        combinations_r = list(combinations(possible_tuples, r))
        all_combinations.extend([list(combo) for combo in combinations_r])

    
    G = return_graph_from_combination(graph)
    G.add_node(new_vertice)

    graphs = [G]
    for combination_graph in all_combinations:
        Gi = G.copy()
        Gi.add_edges_from(combination_graph)
        graphs.append(Gi)
  
    graphs = add_graph_to_list_isomorphics(graphs)

    return all_combinations

def get_mean_automorphism_count_per_edge_number(graph_edge_automorphisms_edgenum_lists):

    edges_list = [graph_edge_automorphisms_edgenum_list[2] for graph_edge_automorphisms_edgenum_list in graph_edge_automorphisms_edgenum_lists]
    min_edge = min(edges_list)
    max_edge = max(edges_list)

    edge_number_mean_count = []

    for edge in range(min_edge,max_edge+1):
         
        graph_with_edge_number = [item for item in graph_edge_automorphisms_edgenum_lists if item[2] == edge]
        graph_with_edge_number_counts = [item[1] for item in graph_with_edge_number]
        sum_with_edge_number =  sum(graph_with_edge_number_counts)
        count = len(graph_with_edge_number)

        edge_number_mean_count.append([edge,sum_with_edge_number/count])
         
    return edge_number_mean_count

def generate_all_possible_connected_graphs_num_edges(n, num_edges, edge_lists = True):
        
        nodes = list(range(n))
        possible_edges = list(combinations(nodes, 2))
        all_graphs_with_num_edges = []
       
        edges_combinations_n_edges = [[edge for edge in combo] for combo in list(combinations(possible_edges, num_edges))]
        graphs_all_nodes = []

        # create graphs from combinations with all nodes/vertices presents
        for combination in edges_combinations_n_edges:
            if set(nodes) <= set(node for edge in combination for node in edge):
                G = return_graph_from_combination(combination, nodes)
                graphs_all_nodes.append(G)

        if edge_lists:
            graphs_all_nodes = [list(graph.edges()) for graph in graphs_all_nodes]

        return graphs_all_nodes

def add_graph_to_list_isomorphics(nx_graphs):
    isomorphic_graphs = []
    for graph in nx_graphs:           
            
        #add to list with weight one if none present are already isomorphic
        if not isomorphic_graphs or not any(nx.is_isomorphic(graph, isomorphic_graph[0]) for isomorphic_graph in isomorphic_graphs): #nx.is_isomorphic is not same type of isomorphism as match.isomorphism_iter generated
                isomorphic_graphs.append([graph,1, graph])
        else:

            isomorphic_to_graph_entries = [existing_graph for existing_graph in isomorphic_graphs if nx.is_isomorphic(graph, existing_graph[0])]
            first_isomorphic_to_graph_entry = isomorphic_to_graph_entries[0][0]
            first_isomorphic_to_graph_entry_weight = isomorphic_to_graph_entries[0][1] +1 #add current graph to weight

            # update isomorphic graph with new weights
            isomorphic_graphs = [
            [element[0],first_isomorphic_to_graph_entry_weight,first_isomorphic_to_graph_entry] if nx.is_isomorphic(graph, element[0]) 
            else element for element in isomorphic_graphs
            ]

            # get number of isomorphic graph already present, get cumulative weight and remove isomorphics
            graph_entry = [graph,first_isomorphic_to_graph_entry_weight,first_isomorphic_to_graph_entry]
            isomorphic_graphs.append(graph_entry)
    
    return isomorphic_graphs

def generate_all_connected_graphs(n_vertices, filter_isomorphics = False): #connected means with no free vertice
    n = n_vertices
    print(f"number of vertices {n} start")
    all_graphs = []

    # Iterate over all possible combinations of edges
    smallest_possible_edge_count = math.ceil(n/2)
    edge_range = range(smallest_possible_edge_count,n*(n-1)//2 + 1)

    for num_edges in edge_range :  # Maximum number of edges for n vertices 
        print(f"edge {num_edges} on {max(edge_range)}")
        graph_with_all_nodes = generate_all_possible_connected_graphs_num_edges(n, num_edges, False)
        # add isomorphic graph and weights
        isomorphic_graphs = []
        for graph in graph_with_all_nodes:           
            
            #add to list with weight one if none present are already isomorphic
            if not isomorphic_graphs or not any(nx.is_isomorphic(graph, isomorphic_graph[0]) for isomorphic_graph in isomorphic_graphs): #nx.is_isomorphic is not same type of isomorphism as match.isomorphism_iter generated
                 isomorphic_graphs.append([graph,1,num_edges, graph])
            else:

                isomorphic_to_graph_entries = [existing_graph for existing_graph in isomorphic_graphs if nx.is_isomorphic(graph, existing_graph[0])]
                first_isomorphic_to_graph_entry = isomorphic_to_graph_entries[0]
                first_isomorphic_to_graph_entry_graph = first_isomorphic_to_graph_entry[3]
                first_isomorphic_to_graph_entry_weight = first_isomorphic_to_graph_entry[1] +1 #add current graph to weight

                # update isomorphic graph with new weights
                isomorphic_graphs = [
                [element[0],first_isomorphic_to_graph_entry_weight,num_edges,first_isomorphic_to_graph_entry_graph] if nx.is_isomorphic(graph, element[0]) 
                else element for element in isomorphic_graphs
                ]

                # get number of isomorphic graph already present, get cumulative weight and remove isomorphics
                graph_entry = [graph,first_isomorphic_to_graph_entry_weight,num_edges,first_isomorphic_to_graph_entry_graph]
                isomorphic_graphs.append(graph_entry)

        all_graphs.extend(isomorphic_graphs)

    if filter_isomorphics:
        all_graphs = [graph for graph in all_graphs if graph[0] == graph[3]] #pick the graph representing to isomorphic group only

    all_graphs_combinations = [[list(graph[0].edges()),graph[1],graph[2],list(graph[3].edges())] for graph in all_graphs]
    return all_graphs_combinations

def generate_isomorphics_from_combination(combination, nodes = None, max_isomorphism_number= None):

    if nodes is None:
        node_list = list(set(node for edge in combination for node in edge))
        max_node_indice = max(node_list)
        nodes = list(range(0,max_node_indice +1))
    
    edge_count = len(combination)

    G = return_graph_from_combination(combination, nodes)
    possible_graphs = generate_all_possible_connected_graphs_num_edges(max_node_indice+1,edge_count)
    
    if max_isomorphism_number is None or max_isomorphism_number == "None":        
        max_isomorphism_number = len(possible_graphs)
    else:
        max_isomorphism_number = min(len(possible_graphs),max_isomorphism_number)

    count = 0
    index = 0
    isomorphic_graph_list = []
    while count < max_isomorphism_number and index < len(possible_graphs):

        G2 = return_graph_from_combination(possible_graphs[index],nodes)
        if nx.is_isomorphic(G, G2):
            count += 1
            isomorphic_graph_list.append(possible_graphs[index])
        index += 1

    return isomorphic_graph_list

def calculate_plot_mean_automorphism_count_per_edge_number(vertices_choice):
    edge_count_lists = []
    for n in vertices_choice: #8
        all_graphs = generate_all_connected_graphs(n)
        edge_meancount = get_mean_automorphism_count_per_edge_number(all_graphs)
        edge_meancount_vertices_number = [item + [str(n) + " vertices"] for item in edge_meancount]
        edge_count_lists.append(edge_meancount_vertices_number)

    plot_xylists(edge_count_lists)

def draw_graph(combination , ax = None):
    G = return_graph_from_combination(combination)
    pos = nx.circular_layout(G)
    nx.draw(G, pos)


def test_combination_list():
    test_combination1 = [(0,1),(0,2),(2,3),(4,3),(4,1)]
    test_combination2 = [(0,1),(0,2),(2,3),(4,2),(4,1)]
    test_combination3 = [(0,1),(0,2),(2,3),(4,1)]
    test_combination31 = [(0,1),(0,2),(2,3),(4,1)]
    test_combination32 = [(0,1),(0,2),(2,3),(4,1)]
    test_combination33 = [(0,1),(0,2),(2,3),(4,1)]
    return [test_combination1,test_combination2,test_combination3,test_combination31,test_combination32,test_combination33]

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


#combos = generate_all_n1_graphs_from_n_graph([(0,1),(1,2)])