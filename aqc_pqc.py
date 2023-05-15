from qiskit.visualization import *
import numpy as np
import scipy.optimize as optimize
import networkx as nx
import collections
from hamiltonian import Hamiltonian
from quantum_circuit import QCir
from qiskit.quantum_info import Statevector


class AQC_PQC():
    def __init__(self, number_of_qubits, problem, steps, layers, single_qubit_gates, entanglement_gates, entanglement, use_third_derivatives = 'No'):

        self.number_of_qubits = number_of_qubits
        self.problem = problem
        self.steps = steps
        self.layers = layers
        self.single_qubit_gates = single_qubit_gates
        self.entanglement_gates = entanglement_gates
        self.entanglement = entanglement

        qcir = QCir(self.number_of_qubits, 'initial', self.layers, self.single_qubit_gates, self.entanglement_gates, self.entanglement)

        self.initial_parameters = qcir.get_initial_parameters()
        self.number_of_parameters = len(self.initial_parameters)

        hamiltonians = Hamiltonian(self.number_of_qubits, self.problem)

        self.offset = hamiltonians.get_offset()
        self.initial_hamiltonian = hamiltonians.construct_initial_hamiltonian()
        self.target_hamiltonian = hamiltonians.construct_problem_hamiltonian()


    def get_expectation_value(self, angles, observable):
        circuit = QCir(self.number_of_qubits, angles, self.layers, self.single_qubit_gates, self.entanglement_gates, self.entanglement)
        sv1 = Statevector.from_label('0'*self.number_of_qubits)
        sv1 = sv1.evolve(circuit.qcir)
        expectation_value = sv1.expectation_value(observable)
        return np.real(expectation_value)
    
    def get_hessian_matrix(self, observable, angles):

        hessian = np.zeros((self.number_of_parameters, self.number_of_parameters))
    
        for parameter1 in range(self.number_of_parameters):
            for parameter2 in range(self.number_of_parameters):
                if parameter1 < parameter2:    
                    
                    hessian_thetas_1, hessian_thetas_2, hessian_thetas_3, hessian_thetas_4 = angles.copy(), angles.copy(), angles.copy(), angles.copy()

                    hessian_thetas_1[parameter1] += np.pi/2
                    hessian_thetas_1[parameter2] += np.pi/2


                    hessian_thetas_2[parameter1] -= np.pi/2
                    hessian_thetas_2[parameter2] += np.pi/2

                    hessian_thetas_3[parameter1] += np.pi/2
                    hessian_thetas_3[parameter2] -= np.pi/2

                    hessian_thetas_4[parameter1] -= np.pi/2
                    hessian_thetas_4[parameter2] -= np.pi/2

                    hessian[parameter1, parameter2] += self.get_expectation_value(hessian_thetas_1, observable)/4
                    hessian[parameter1, parameter2] -= self.get_expectation_value(hessian_thetas_2, observable)/4
                    hessian[parameter1, parameter2] -= self.get_expectation_value(hessian_thetas_3, observable)/4
                    hessian[parameter1, parameter2] += self.get_expectation_value(hessian_thetas_4, observable)/4

                    hessian[parameter2, parameter1] = hessian[parameter1, parameter2]
                    
                if parameter1 == parameter2:

                    hessian_thetas_1 , hessian_thetas_2 = angles.copy(), angles.copy()

                    hessian_thetas_1[parameter1] += np.pi
                    hessian_thetas_2[parameter1] -= np.pi
                    
                    hessian[parameter1, parameter1] += self.get_expectation_value(hessian_thetas_1, observable)/4
                    hessian[parameter1, parameter1] += self.get_expectation_value(hessian_thetas_2, observable)/4
                    hessian[parameter1, parameter1] -= self.get_expectation_value(angles, observable)/2

        return hessian

    
    def get_derivative(self, observable, which_parameter, parameters):

        derivative = 0
        parameters_plus, parameters_minus = parameters.copy(), parameters.copy()
        parameters_plus[which_parameter] += np.pi
        parameters_minus[which_parameter] -= np.pi

        derivative += 1/2*self.get_expectation_value(parameters_plus, observable)
        derivative -= 1/2*self.get_expectation_value(parameters_minus, observable)

        return derivative

    def get_instantaneous_hamiltonian(self, time):
        return (1-time)*self.initial_hamiltonian + time*self.target_hamiltonian
    
    def get_linear_system(self, hamiltonian, angles): #Construct function get_derivatives() and replace zero order terms. Also replace below with get_hessian

        zero_order_terms = np.zeros((self.number_of_parameters,))
        first_order_terms = np.zeros((self.number_of_parameters, self.number_of_parameters))

        #We start with zero order terms.
        for parameter in range(self.number_of_parameters):
            zero_order_terms[parameter] += self.get_derivative(hamiltonian, parameter, angles)

        first_order_terms = self.get_hessian_matrix(hamiltonian, angles)

        return np.array(zero_order_terms), np.array(first_order_terms)
    
    def minimum_eigenvalue(self, matrix):

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        min_eigen = np.min(eigenvalues)
        print(min_eigen)
        return min_eigen

    def run(self):
        
        energies_aqcpqc = []

        lambdas = [i for i in np.linspace(0, 1, self.steps+1)][1:]
        optimal_thetas = self.initial_parameters.copy()
        print(f'We start with the optimal angles of the initial hamiltonian: {optimal_thetas}')

        initial_hessian = self.get_hessian_matrix(self.initial_hamiltonian, optimal_thetas) 
        w, v = np.linalg.eig(initial_hessian)
        print(f'The eigenvalues of the initial Hessian are {np.round(w, 7)}')

        for lamda in lambdas:
            print('\n')
            print(f'We are working on {lamda}')
            hamiltonian = self.get_instantaneous_hamiltonian(lamda)
            zero, first = self.get_linear_system(hamiltonian, optimal_thetas)

            def equations(x):
                array = np.array([x[_] for _ in range(self.number_of_parameters)])
                equations = zero + first@array

                y = np.array([equations[_] for _ in range(self.number_of_parameters)])
                return y@y


            def minim_eig_constraint(x):
                new_thetas = [optimal_thetas[i] + x[i] for i in range(self.number_of_parameters)]
                return self.minimum_eigenvalue(self.get_hessian_matrix(hamiltonian, new_thetas))

            cons = [{'type': 'ineq', 'fun':minim_eig_constraint}]
            res = optimize.minimize(equations, x0 = [0 for _ in range(self.number_of_parameters)], constraints=cons,  method='SLSQP',  options={'disp': True, 'maxiter':700}) 
            epsilons = [res.x[_] for _ in range(self.number_of_parameters)]
            
            
            print(f'The solutions of equations are {epsilons}')
            optimal_thetas = [optimal_thetas[_] + epsilons[_] for _ in range(self.number_of_parameters)]

            hessian = self.get_hessian_matrix(hamiltonian, optimal_thetas)
            min_eigen = self.minimum_eigenvalue(hessian)


            inst_exp_value = self.get_expectation_value(optimal_thetas, hamiltonian) - lamda*self.offset
            energies_aqcpqc.append(inst_exp_value)

            print(f'and the minimum eigenvalue of the Hessian at the solution is {min_eigen}')
            print(f'and the instantaneous expectation values is {inst_exp_value}')

        return energies_aqcpqc

