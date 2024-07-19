import pennylane as qml
from pennylane import numpy as np
import math
from QAOA_utils import *
from graph_methods import *
from collections import Counter
from collections import namedtuple
from scipy.optimize import minimize

# from .adagrad import AdagradOptimizer
# from .adam import AdamOptimizer
# from .adaptive import AdaptiveOptimizer
# from .gradient_descent import GradientDescentOptimizer
# from .momentum import MomentumOptimizer
# from .nesterov_momentum import NesterovMomentumOptimizer
# from .riemannian_gradient import RiemannianGradientOptimizer
# from .rms_prop import RMSPropOptimizer
# from .qng import QNGOptimizer
# from .rotosolve import RotosolveOptimizer
# from .rotoselect import RotoselectOptimizer
# from .shot_adaptive import ShotAdaptiveOptimizer
# from .spsa import SPSAOptimizer
# from .qnspsa import QNSPSAOptimizer


# JobResult = namedtuple('JobResult', ['cost_layer', 'label','graph','most_sampled_value','most_sampled_value_ratio','mean','maximum','stdev','parameters'])
QAOAResult = namedtuple(
    'QAOAResult', ['bit_strings_objectives_distribution', 'parameters'])


def qaoa_maxcut(graph, n_wires, n_layers, cost_layer="QAOA", n_steps=30, n_samples=200, lightning_device=True, mixer_layer="fermionic_Ryy", label=None):
    # [isomorph_graph,n_vertices, n_layers, "QAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples]
    # initialize the parameters near zero

    # new parameters for fQAOA and QAOA
    # row 1: angle 2-1 difference for fRyy entangling gate on mixer layer
    # no parameter for cost layer

    # minimize the negative of the objective function
    def objective(params):  # only params to be optimized here for optimizer fct
        thetas1 = params[0]
        thetas2 = params[1]
        neg_obj = 0

        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - qml.QNode(lambda thetas1, thetas2: circuit(
                graph, cost_layer, n_wires, thetas1, thetas2, edge=edge, n_layers=n_layers), dev)(thetas1, thetas2))

        return neg_obj

    def objective_flat(params_flat):
        return objective(params_flat.reshape((2, n_layers)))

    def optimize(optimizer="BFGS", n_steps=30):
        optimized_parameters = []
        init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

        if optimizer == "BFGS":
            # scipy optimize minimize
            result = minimize(
                objective_flat, init_params.flatten(), options={'maxiter': 100*n_layers})

            optimized_params_flat = result.x
            optimized_parameters = optimized_params_flat.reshape((2, n_layers))

        else:  # original QAOA implementation
            # initialize optimizer: Adagrad works well empirically
            opt = qml.AdagradOptimizer(stepsize=0.5)
            print("adagraph optimizer")
            for i in range(n_steps):
                optimized_parameters = init_params
                optimized_parameters = opt.step(objective, init_params)
                print(f"        optim. step {i} on {n_steps} for {label}")

        return optimized_parameters

    if lightning_device:
        dev = qml.device("lightning.qubit", wires=n_wires, shots=1)
    else:
        dev = qml.device("default.qubit", shots=1)

    optimized_parameters = optimize(n_steps=n_steps, optimizer="BFGS")

    bit_strings = sample_bitstrings(
        graph, cost_layer, n_wires, thetas1=optimized_parameters[0], thetas2=optimized_parameters[1], n_samples=n_samples, n_layers=n_layers)

    bitstring_counter = Counter([tuple(arr) for arr in bit_strings])
    objective_counter = [(bitstring_to_objective(key, graph), value)
                         for key, value in bitstring_counter.items()]

    string_params = param_to_string(optimized_parameters)

    return QAOAResult(objective_counter, string_params)


def sample_bitstrings(graph, cost_layer, n_wires, thetas1, thetas2, n_samples, n_layers=1, lightning_device=True):

    if lightning_device:
        dev = qml.device("lightning.qubit", wires=n_wires, shots=1)
    else:
        dev = qml.device("default.qubit", shots=1)

    bit_strings = []

    for _ in range(n_samples):
        bit_string = circuit_samples(
            graph, cost_layer, n_wires, thetas1, thetas2, n_layers=n_layers)
        bit_string = [int(result == 1) for result in bit_string]
        bit_strings.append(bit_string)

    return bit_strings


def circuit(graph, cost_layer, n_wires, thethas1, thethas2, edge=None, n_layers=1):

    # one mixer application to create superposition
    U_B(graph, n_wires, math.pi/2, math.pi/2)

    # p instances of unitary operators
    for i in range(n_layers):
        U_C(graph, option=cost_layer)
        U_B(graph, n_wires, thethas1[i], thethas2[i])

    # during the optimization phase we are evaluating a term
    # in the objective using expval
    H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    return qml.expval(H)


def circuit_samples(graph, cost_layer, n_wires, thethas1, thethas2, n_layers=1, lightning_device=True):

    if lightning_device:
        dev_sample = qml.device("lightning.qubit", wires=n_wires, shots=1)
    else:
        dev_sample = qml.device("default.qubit", shots=1)

    @ qml.qnode(dev_sample)
    def quantum_circuit(thethas1, thethas2):
        # Apply the unitary operations
        # create an initial superposition
        U_B(graph, n_wires, math.pi/2, math.pi/2)

        for i in range(n_layers):
            U_C(graph, option=cost_layer)
            U_B(graph, n_wires, thethas1[i], thethas2[i])

        measurement_values = [qml.expval(qml.PauliZ(w))
                              for w in range(n_wires)]

        return measurement_values

    results = quantum_circuit(thethas1, thethas2)

    return results

# mixer layer


def U_B(graph, n_wires, theta1, theta2):
    # fRyy(pi/2,pi/2) for next qbits mixer
    for wire in range(n_wires-1):
        qml.QubitUnitary(fRyy(theta1, theta2), wires=[wire, wire+1])

# cost layer


def U_C(graph, option="QAOA"):
    # def U_C(graph, beta, option="QAOA"):

    if option == "QAOA":  # regular QAOA algorithm cost layer
        for edge in graph:
            wire1 = edge[0]
            wire2 = edge[1]
            qml.CNOT(wires=[wire1, wire2])
            qml.RZ(math.pi, wires=wire2)  # qml.RZ(beta, wires=wire2)
            qml.CNOT(wires=[wire1, wire2])

    elif option == "swapQAOA":  # regular QAOA algorithm cost layer but with swap up to neighbor operation
        for edge in graph:

            # pick the edge with smallest index
            first_edge = edge[0]
            last_edge = edge[1]

            if edge[1] == edge[0]:
                return  # invalid graph, j=k
            if edge[0] > edge[1]:
                last_edge == edge[0]
                first_edge = edge[1]

            # apply swaps from the first edge to the last edge
            for vertice in range(first_edge, last_edge):
                qml.SWAP(wires=[vertice, vertice+1])

            # apply gate to adjacent swapped qbits
            qml.CNOT(wires=[vertice, vertice+1])
            qml.RZ(math.pi, wires=vertice+1)
            qml.CNOT(wires=[vertice, vertice+1])

            # apply swaps back fromt the last edge to the first edge
            for vertice in reversed(range(first_edge, last_edge)):
                qml.SWAP(wires=[vertice, vertice+1])

    elif option == "fQAOA":  # fermi QAOA algorithm cost layer with fswaps and fixed angle = pi for Rzz
        for edge in graph:

            # pick the edge with smallest index
            first_edge = edge[0]
            last_edge = edge[1]

            if edge[1] == edge[0]:
                return  # invalid graph, j=k
            if edge[0] > edge[1]:
                last_edge == edge[0]
                first_edge = edge[1]

            # apply fswaps from the first edge to the last edge
            for vertice in range(first_edge, last_edge):
                qml.QubitUnitary(fSwap, wires=[vertice, vertice+1])

            qml.CNOT(wires=[last_edge-1, last_edge])
            qml.RZ(math.pi, wires=last_edge)  # qml.RZ(beta, wires=wire2)
            qml.CNOT(wires=[last_edge-1, last_edge])

            # apply gate to adjacent swapped qbits
            # qml.QubitUnitary(Rzz_matrice(fixed_beta),
            #                  wires=[last_edge-1, last_edge])

            # apply fswaps back from the last edge to the first edge
            for vertice in reversed(range(first_edge, last_edge)):
                qml.QubitUnitary(fSwap, wires=[vertice, vertice+1])
