import pennylane as qml
from pennylane import numpy as np
import math
import cmath
from QAOA_utils import *
from graph import *
from collections import Counter
from collections import namedtuple

QAOAResult = namedtuple('QAOAResult', ['bit_strings_objectives_distribution', 'parameters'])

# connected  graphs with n nodes:

#Nodes     2   3    4      5        6          7            8              9
#Unlabeled 1,  2,   6,    21,     112,       853,       11117,         261080
#Labeled   4, 38, 728, 26704, 1866256, 251548592, 66296291072, 34496488594816

def execute_qaoa_subjob1(graph,n_vertices, n_layers, cost_layer, label, n_steps = 30, n_samples = 200):
    np.random.seed(42)
    start_time = time.time()
    graph_results = qaoa_maxcut(n_vertices,n_layers, graph,n_vertices, cost_layer=cost_layer , n_steps = n_steps, n_samples = n_samples) #n_layer = n_vertices
    graph_results_distribution, graph_results_parameters  = graph_results.bit_strings_objectives_distribution, graph_results.parameters
    most_common_element, most_common_element_count_ratio, mean, maximum, stdev = compute_stats(graph_results_distribution)

    elapsed_time_seconds = time.time() - start_time
    elapsed_time_formatted = f"{int(elapsed_time_seconds // 60)} mins {int(elapsed_time_seconds % 60)} secs"

    #chi squared
    return [cost_layer,label, graph_to_string(graph), most_common_element, most_common_element_count_ratio, mean, maximum, stdev, str(graph_results_parameters)]

def qaoa_maxcut(n_wires, n_layers, graph, mixer_layer = "fermionic_Ryy", cost_layer = "QAOA", n_steps = 30, n_samples = 200):

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    dev = qml.device("lightning.qubit", wires=n_wires, shots=1)

    # minimize the negative of the objective function
    def objective(params): #only params to be optimized here for optimizer fct
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - qml.QNode(lambda gammas, betas: circuit(graph, n_wires, gammas, betas, edge=edge, n_layers=n_layers), dev)(gammas, betas)) #jew 2

        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params

    for i in range(n_steps):
        params = opt.step(objective, params)
        # if (i + 1) % 5 == 0:
        #     print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))
                   
    # sample measured bitstrings 100 times
    bit_strings = sample_bitstrings(graph,n_wires,gammas=params[0], betas = params[1], n_samples= n_samples, n_layers= n_layers)
   
    bitstring_counter = Counter([tuple(arr) for arr in bit_strings])
    objective_counter = [(bitstring_to_objective(key, graph), value) for key, value in bitstring_counter.items()]

    return QAOAResult(objective_counter, params)

def sample_bitstrings(graph, n_wires, gammas, betas, n_samples, n_layers=1):

    dev = qml.device("lightning.qubit", wires=n_wires, shots=1)

    bit_strings = []

    for _ in range(n_samples):
        bit_string = circuit_samples(graph, n_wires, gammas, betas, n_layers=n_layers)
        bit_string = [int(result == 1) for result in bit_string]
        bit_strings.append(bit_string)

    return bit_strings

def circuit(graph, n_wires, gammas, betas, edge=None, n_layers=1):

    #one mixer application instead of hadamard gates
    U_B(graph, n_wires, gammas[0])

    # p instances of unitary operators
    for i in range(n_layers):
        U_C(graph,gammas[i])
        U_B(graph,n_wires, betas[i])

    # during the optimization phase we are evaluating a term
    # in the objective using expval
    H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    return qml.expval(H)

def circuit_samples(graph, n_wires, gammas, betas, n_layers=1):

    dev_sample = qml.device("lightning.qubit", wires=n_wires, shots=1)

    @qml.qnode(dev_sample)
    def quantum_circuit(gammas, betas):
        # Apply the unitary operations
        U_B(graph, n_wires, gammas[0])
        for i in range(n_layers):
            U_C(graph, gammas[i])
            U_B(graph, n_wires, betas[i])


        measurement_values = [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

        return measurement_values


    results = quantum_circuit(gammas, betas)

    return results

# mixer layer
def U_B(graph, n_wires, beta):    
    #fRyy(pi/2,pi/2) for next qbits mixer (Jasmin)
    for wire in range(n_wires-1):
        qml.QubitUnitary(fRyy, wires=[wire, wire+1])

# cost layer
def U_C(graph, gamma, option = "QAOA"):

    #set Gamma = pi/2
    fixed_gamma = math.pi  #todo: check pi/2

    if option == "QAOA": # regular QAOA algorithm  cost layer
        for edge in graph:               
            wire1 = edge[0]
            wire2 = edge[1]
            qml.CNOT(wires=[wire1, wire2])
            qml.RZ(gamma, wires=wire2)
            qml.CNOT(wires=[wire1, wire2])

    elif option == "swapQAOA": # regular QAOA algorithm cost layer but with swap
        for edge in graph:

            #pick the edge with smallest index
            first_edge = edge[0]
            last_edge = edge[1]

            if edge[1] == edge[0]:
                return #invalid graph, j=k
            if edge[0] > edge[1]:
                last_edge == edge[0]
                first_edge = edge[1]

            #apply swaps from the first edge to the last edge
            for vertice in range(first_edge, last_edge):                                 
                qml.SWAP(wires=[vertice, vertice+1])

            #apply gate to adjacent swapped qbits                
            qml.CNOT(wires=[vertice, vertice+1])
            qml.RZ(gamma, wires=vertice+1)
            qml.CNOT(wires=[vertice, vertice+1])

            #apply swaps back fromt the last edge to the first edge
            for vertice in reversed(range(first_edge, last_edge)):                          
                qml.SWAP(wires=[vertice, vertice+1])
    
    elif option == "fQAOA": # fermi QAOA algorithm cost layer with fswaps
        for edge in graph:

            #pick the edge with smallest index
            first_edge = edge[0]
            last_edge = edge[1]

            if edge[1] == edge[0]:
                return #invalid graph, j=k
            if edge[0] > edge[1]:
                last_edge == edge[0]
                first_edge = edge[1]

            #apply fswaps from the first edge to the last edge
            for vertice in range(first_edge, last_edge):                          
                qml.QubitUnitary(fSwap,wires=[vertice, vertice+1])

            #apply gate to adjacent swapped qbits                
            qml.QubitUnitary(Rzz_matrice(fixed_gamma), wires=[last_edge-1, last_edge])

            #apply fswaps back fromt the last edge to the first edge
            for vertice in reversed(range(first_edge, last_edge)):                          
                qml.QubitUnitary(fSwap,wires=[vertice, vertice+1])
