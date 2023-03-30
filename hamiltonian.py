import numpy as np


class Hamiltonian():

    def __init__(self, number_of_qubits, problem):
        
        self.number_of_qubits = number_of_qubits
        self.pauliz = np.array([[1, 0], [0, -1]])
        self.paulix = np.array([[0, 1], [1, 0]])
        self.pauliy = np.array([[0, -1j], [1j, 0]])
        self.identity = np.array([[1, 0], [0, 1]])
        self.problem_type = problem['type']
        self.problem_properties = problem['properties']

    def tensor_pauli(self, which_qubit, pauli_matrix):

        if which_qubit == 0:
            matrix = pauli_matrix
        else:
            matrix = self.identity

        for qubit in range(1, self.number_of_qubits):
            if which_qubit == qubit:
                matrix = np.kron(pauli_matrix, matrix)
            else:
                matrix = np.kron(self.identity, matrix)

        return matrix
    
    def construct_initial_hamiltonian(self):

        initial_Hamiltonian = np.zeros((2**self.number_of_qubits, 2**self.number_of_qubits))
        for qubit in range(self.number_of_qubits):
            initial_Hamiltonian -= self.tensor_pauli(qubit, self.paulix)

        return initial_Hamiltonian

    def construct_problem_hamiltonian(self): #I could put problem as a dictionary problem = {'type' = MaxCut, [list_of_things_to_define_problem] = w}

        Hamiltonian = np.zeros((2**self.number_of_qubits, 2**self.number_of_qubits))

        if self.problem_type == 'MaxCut':
            adjacency_matrix = self.problem_properties

            for vertex1 in range(self.number_of_qubits):
                for vertex2 in range(self.number_of_qubits):
                    if vertex1 < vertex2:
                        if adjacency_matrix[vertex1, vertex2] != 0:
                            Hamiltonian += 1/2*adjacency_matrix[vertex1, vertex2]*self.tensor_pauli(vertex1, self.pauliz)@self.tensor_pauli(vertex2, self.pauliz)

        elif self.problem_type == 'NumberPartitioning':
            numbers_list = self.problem_properties
            for qubit1 in range(self.number_of_qubits):
                for qubit2 in range(self.number_of_qubits):
                    if qubit1<qubit2:
                        Hamiltonian += 2*numbers_list[qubit1]*numbers_list[qubit2]*self.tensor_pauli(qubit1, self.pauliz)@self.tensor_pauli(qubit2, self.pauliz)


        elif self.problem_type == 'TFIC':
            jays, h = self.problem_properties[0], self.problem_properties[1]

            for qubit in range(self.number_of_qubits-1):
                Hamiltonian -= jays[qubit]*self.tensor_pauli(qubit, self.pauliz)@self.tensor_pauli(qubit+1, self.pauliz)
            Hamiltonian -= jays[self.number_of_qubits-1]*self.tensor_pauli(self.number_of_qubits-1, self.pauliz)@ self.tensor_pauli(0, self.pauliz)
            
            for qubit in range(self.number_of_qubits):
                Hamiltonian -= h*self.tensor_pauli(qubit, self.paulix)


        return Hamiltonian
    
    def get_offset(self):
        offset = 0


        if self.problem_type == 'MaxCut':

            adjacency_matrix = self.problem_properties
            for i in range(self.number_of_qubits):
                for j in range(self.number_of_qubits):
                    if i<j:
                        offset += adjacency_matrix[i,j]/2

        elif self.problem_type == 'NumberPartitioning':
            
            numbers_list = self.problem_properties
            for num in numbers_list:
                offset += num**2

        return offset

