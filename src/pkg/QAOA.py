import pennylane as qml
from pennylane import numpy as np
import math
import cmath
from QAOA_utils import *
from graph import *
from collections import Counter
from collections import namedtuple

graph = "default graph" #set in the main file
n_samples = 200 # 100, 500 set in main file
steps = 30 # set in main file
n_layers = 4 #set in main file
QAOAResult = namedtuple('QAOAResult', ['bit_strings_objectives_distribution', 'parameters'])

# unitary operator U_B with parameter beta
def U_B(graph, n_wires, beta):
    
    #fRyy(pi/2,pi/2) for next qbits mixer (Jasmin)
    for wire in range(n_wires-1):
        qml.QubitUnitary(fRyy, wires=[wire, wire+1])

    # G(H,H) for qbit pairs mixer (Marco)
    #for wire in range(n_wires)[::2]:
    #    qml.QubitUnitary(fRhh, wires=[wire, wire+1])
    
    # original mixer
    #for wire in range(n_wires):
    #    qml.RX(2 * beta, wires=wire)

# unitary operator U_C with parameter gamma
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

#We also require a quantum node which will apply the operators according to the angle parameters, and return the expectation 
#value of the observable σjzσkz to be used in each term of the objective function later on. The argument edge specifies the 
# chosen edge term in the objective function, (j,k). Once optimized, the same quantum node can be used for sampling
# an approximately optimal bitstring if executed with the edge keyword set to None. Additionally, we specify the number of layers 
#(repeated applications of UBUC) using the keyword n_layers.

dev = qml.device("default.qubit", shots=1)
@qml.qnode(dev)
def circuit(graph, n_wires, gammas, betas, edge=None, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    #for wire in range(n_wires):
    #    qml.Hadamard(wires=wire)

    #one mixer application instead of hadamard gates
    U_B(graph, n_wires, gammas[0])

    # p instances of unitary operators
    for i in range(n_layers):
        U_C(graph,gammas[i])
        U_B(graph,n_wires, betas[i])
    if edge is None:
        # measurement phase
        return qml.sample()
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    return qml.expval(H)


def qaoa_maxcut(n_wires, graph, mixer_layer = "fermionic_Ryy", cost_layer = "QAOA"):

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params): #only params to be optimized here for optimizer fct
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(graph, n_wires, gammas, betas, edge=edge, n_layers=n_layers))
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params

    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))
                   
    # sample measured bitstrings 100 times
    bit_strings = []

    for i in range(0, n_samples):
        bit_strings.append(circuit(graph,n_wires, params[0], params[1], edge=None, n_layers=n_layers))
    
    bitstring_counter = Counter([tuple(arr) for arr in bit_strings])
    objective_counter = [(bitstring_to_objective(key, graph), value) for key, value in bitstring_counter.items()]

    return QAOAResult(objective_counter, params)
    #return -objective(params), bit_strings, objectives, params


# connected  graphs with n nodes:

#Nodes     2   3    4      5        6          7            8              9
#Unlabeled 1,  2,   6,    21,     112,       853,       11117,         261080
#Labeled   4, 38, 728, 26704, 1866256, 251548592, 66296291072, 34496488594816


