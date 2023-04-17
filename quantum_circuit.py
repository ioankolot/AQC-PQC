from qiskit import QuantumCircuit
from qiskit.visualization import *
from qiskit.circuit.library import TwoLocal
import numpy as np
from qiskit.quantum_info import Statevector


class QCir():
    def __init__(self, number_of_qubits, thetas, layers, single_qubit_gates, entanglement_gates, entanglement):

        self.number_of_qubits = number_of_qubits
        self.qcir = QuantumCircuit(self.number_of_qubits)
        self.single_qubit_gates = single_qubit_gates
        self.entanglement_gates = entanglement_gates
        self.layers = layers


        self.qcir += TwoLocal(num_qubits = number_of_qubits, rotation_blocks = single_qubit_gates, entanglement_blocks = entanglement_gates, reps = layers, entanglement=entanglement)
        self.number_of_parameters = self.qcir.num_parameters 
        #print(self.number_of_parameters)

        if thetas == 'initial':
            thetas = self.get_initial_parameters()

        self.qcir = self.qcir.assign_parameters(thetas)



    def get_initial_parameters(self):

        initial_parameters = []

        if self.single_qubit_gates == 'ry' and self.entanglement_gates == 'cz':
            for qubit in range(self.number_of_qubits*(self.layers)):
                initial_parameters.append(0)

            for qubit in range(self.number_of_qubits):
                initial_parameters.append(np.pi/2)

        

        return initial_parameters 
    
    def calculate_expectation_value(self, matrix): #This function calculates the expectation value of a given observable (given as a matrix)
        statevector = Statevector.from_label('0'*self.number_of_qubits)
        statevector = statevector.evolve(self.qcir)
        expectation_value = statevector.expectation_value(matrix)
        return expectation_value
