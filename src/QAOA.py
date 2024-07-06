import pennylane as qml
from pennylane import numpy as np
import math
from QAOA_utils import *
from graph_methods import *
from collections import Counter
from collections import namedtuple

#JobResult = namedtuple('JobResult', ['cost_layer', 'label','graph','most_sampled_value','most_sampled_value_ratio','mean','maximum','stdev','parameters'])
QAOAResult = namedtuple('QAOAResult', ['bit_strings_objectives_distribution', 'parameters'])

def qaoa_maxcut(graph, n_wires, n_layers, cost_layer = "QAOA", n_steps = 30, n_samples = 200, lightning_device = True, mixer_layer = "fermionic_Ryy"):
    #[isomorph_graph,n_vertices, n_layers, "QAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples]
    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    if lightning_device:
        dev = qml.device("lightning.qubit", wires=n_wires, shots=1)
    else: 
        dev = qml.device("default.qubit",shots=1)

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
                   
    # sample measured bitstrings 100 times
    bit_strings = sample_bitstrings(graph,n_wires,gammas=params[0], betas = params[1], n_samples= n_samples, n_layers= n_layers)
   
    bitstring_counter = Counter([tuple(arr) for arr in bit_strings])
    objective_counter = [(bitstring_to_objective(key, graph), value) for key, value in bitstring_counter.items()]

    return QAOAResult(objective_counter, params)

def sample_bitstrings(graph, n_wires, gammas, betas, n_samples, n_layers=1, lightning_device = True):

    if lightning_device:
        dev = qml.device("lightning.qubit", wires=n_wires, shots=1)
    else: 
        dev = qml.device("default.qubit",shots=1)

    bit_strings = []

    for _ in range(n_samples):
        bit_string = circuit_samples(graph, n_wires, gammas, betas, n_layers=n_layers)
        bit_string = [int(result == 1) for result in bit_string]
        bit_strings.append(bit_string)

    return bit_strings

def circuit(graph, n_wires, gammas, betas, edge=None, n_layers=1):

    #one mixer application to create superposition
    U_B(graph, n_wires, gammas[0])

    # p instances of unitary operators
    for i in range(n_layers):
        U_C(graph,gammas[i])
        U_B(graph,n_wires, betas[i])

    # during the optimization phase we are evaluating a term
    # in the objective using expval
    H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    return qml.expval(H)

def circuit_samples(graph, n_wires, gammas, betas, n_layers=1, lightning_device = True):

    if lightning_device:
        dev_sample = qml.device("lightning.qubit", wires=n_wires, shots=1)
    else: 
        dev_sample = qml.device("default.qubit",shots=1)

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
    #fRyy(pi/2,pi/2) for next qbits mixer 
    for wire in range(n_wires-1):
        qml.QubitUnitary(fRyy, wires=[wire, wire+1])

# cost layer
def U_C(graph, gamma, option = "QAOA"):

    fixed_gamma = math.pi  

    if option == "QAOA": # regular QAOA algorithm cost layer
        for edge in graph:               
            wire1 = edge[0]
            wire2 = edge[1]
            qml.CNOT(wires=[wire1, wire2])
            qml.RZ(gamma, wires=wire2)
            qml.CNOT(wires=[wire1, wire2])

    elif option == "swapQAOA": # regular QAOA algorithm cost layer but with swap up to neighbor operation
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
