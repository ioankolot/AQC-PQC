from aqc_pqc import AQC_PQC
from hamiltonian import Hamiltonian 
from quantum_circuit import QCir
import networkx as nx
from brute_force import Brute_Force
from qaoa_circuit import QAOA_Circuit
from aqc_qaoa import AQC_PQC_QAOA
import numpy as np

np.random.seed(2)
seed = 3
number_of_qubits = 4
steps = 100 #Choose number of steps to interpolate from initial to final Hamiltonian
connectivity = 'nearest-neighbors' #This is the connectivity of the non-parameterized gates in the Hardware Efficient ansatz
single_qubit_gates = 'ry'
entanglement_gates = 'cz'
layers = 1
entanglement = 'linear'

graph = nx.random_regular_graph(3, number_of_qubits, seed=seed)
w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))

problem = {'type':'MaxCut', 'properties': w}
Brute_Force(problem)


#aqc_pqc = AQC_PQC_QAOA(number_of_qubits, problem, steps, layers) #Uncomment if you want to use QAOA ansatz.
#aqc_pqc.run()

aqc_pqc = AQC_PQC(number_of_qubits, problem, steps, layers, single_qubit_gates,
                  entanglement_gates, entanglement, use_null_space=False, use_third_derivatives=True)
aqc_pqc.run()


