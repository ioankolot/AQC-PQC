import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.applications import Maxcut, NumberPartition

class Hamiltonian():


    def __init__(self, problem):
        self.problem_type = problem['type']
        self.problem_properties = problem['properties']


    def get_pauli_operator_and_offset(self):
        
        if self.problem_type == 'MaxCut':
            
            w = self.problem_properties
            maxcut = Maxcut(w)
            qp = maxcut.to_quadratic_program()
            qubitOp, offset = qp.to_ising()


        elif self.problem_type == 'Number_Partition':
            
            number_list = self.problem_properties
            num_par = NumberPartition(number_list)
            qp = num_par.to_quadratic_program()
            qubitOp, offset = qp.to_ising()

        return qubitOp, offset
    
    def get_transverse_hamiltonian(self, number_of_qubits):
        X_tuples = []
        
        for i in range(number_of_qubits):
            X_tuples.append(('X', [i], -1))

        hamiltonian = SparsePauliOp.from_sparse_list([*X_tuples], num_qubits = number_of_qubits)
        
        return hamiltonian
