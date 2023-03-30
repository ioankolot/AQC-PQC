from aqc_pqc import AQC_PQC
from hamiltonian import Hamiltonian 
from quantum_circuit import QCir
import networkx as nx


seed = 1
number_of_qubits = 4
steps = 30 #Choose number of steps to interpolate from initial to final Hamiltonian
connectivity = 'nearest-neighbors' #This is the connectivity of the non-parameterized gates in the Hardware Efficient ansatz
single_qubit_gates = 'ry'
entanglement_gates = 'cz'
layers = 1
entanglement = 'linear'

graph = nx.random_regular_graph(3, number_of_qubits, seed=1)
w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))

problem = {'type':'MaxCut', 'properties': w}


aqc_pqc = AQC_PQC(number_of_qubits, problem, steps, layers, single_qubit_gates, entanglement_gates, entanglement)
aqc_pqc.run()