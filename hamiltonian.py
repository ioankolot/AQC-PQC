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

        elif self.problem == 'NumberPartitioning':
            numbers_list = self.problem_properties
            pass

        elif self.problem == 'TFIC':
            pass

        return Hamiltonian