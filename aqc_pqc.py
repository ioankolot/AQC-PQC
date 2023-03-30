from qiskit.visualization import *
import numpy as np
import scipy.optimize as optimize
import networkx as nx
import collections
from hamiltonian import Hamiltonian
from quantum_circuit import QCir
from qiskit.quantum_info import Statevector

#is constructing the circuit and assigning parameters harder than running the circuit all the time

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
    

    def get_instantaneous_hamiltonian(self, time):
        return (1-time)*self.initial_hamiltonian + time*self.target_hamiltonian
    
    def get_linear_system(self, hamiltonian, angles): #Construct function get_derivatives() and replace zero order terms. Also replace below with get_hessian

        zero_order_terms = np.zeros((self.number_of_parameters,))
        first_order_terms = np.zeros((self.number_of_parameters, self.number_of_parameters))

        #We start with zero order terms.
        for parameter in range(self.number_of_parameters):

            zero_order_thetas_1, zero_order_thetas_2 = angles.copy(), angles.copy()
            zero_order_thetas_1[parameter] += np.pi/2
            zero_order_thetas_2[parameter] -= np.pi/2


            zero_order_terms[parameter] += 1/2*self.get_expectation_value(zero_order_thetas_1, hamiltonian)
            zero_order_terms[parameter] -= 1/2*self.get_expectation_value(zero_order_thetas_2, hamiltonian)

        #Next we continue with first order terms.
        for parameter1 in range(self.number_of_parameters):
            for parameter2 in range(self.number_of_parameters):
                if parameter1 <= parameter2:
                
                    first_order_thetas_1, first_order_thetas_2, first_order_thetas_3, first_order_thetas_4 = angles.copy(), angles.copy(), angles.copy(), angles.copy()

                    first_order_thetas_1[parameter1] += np.pi/2
                    first_order_thetas_1[parameter2] += np.pi/2

                    first_order_thetas_2[parameter1] += np.pi/2
                    first_order_thetas_2[parameter2] -= np.pi/2

                    first_order_thetas_3[parameter1] -= np.pi/2
                    first_order_thetas_3[parameter2] += np.pi/2

                    first_order_thetas_4[parameter1] -= np.pi/2
                    first_order_thetas_4[parameter2] -= np.pi/2

                    first_order_terms[parameter1, parameter2] += self.get_expectation_value(first_order_thetas_1, hamiltonian)/4
                    first_order_terms[parameter1, parameter2] -= self.get_expectation_value(first_order_thetas_2, hamiltonian)/4
                    first_order_terms[parameter1, parameter2] -= self.get_expectation_value(first_order_thetas_3, hamiltonian)/4
                    first_order_terms[parameter1, parameter2] += self.get_expectation_value(first_order_thetas_4, hamiltonian)/4

                    first_order_terms[parameter2, parameter1] = first_order_terms[parameter1, parameter2]

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
        print(f'The eigenvalues of the initial Hessian are {np.round(w,5)}')

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
            res = optimize.minimize(equations, x0 = [0 for _ in range(self.number_of_parameters)], constraints=cons,  method='COBYLA',  options={'disp': True, 'maxiter':700}) 
            epsilons = [res.x[_] for _ in range(self.number_of_parameters)]
            
            
            print(f'The solutions of equations are {epsilons}')
            optimal_thetas = [optimal_thetas[_] + epsilons[_] for _ in range(self.number_of_parameters)]

            hessian = self.get_hessian_matrix(hamiltonian, optimal_thetas)
            min_eigen = self.minimum_eigenvalue(hessian)


            inst_exp_value = self.get_expectation_value(optimal_thetas, hamiltonian) #- lamda*offset
            energies_aqcpqc.append(inst_exp_value)

            print(f'and the minimum eigenvalue of the Hessian at the solution is {min_eigen}')
            print(f'and the instantaneous expectation values is {inst_exp_value}')

        return energies_aqcpqc



'''
def best_cost_brute(adjacency_matrix): #This function calculates the optimal cost function by brute force
    best_cost = 0
    number_of_qubits = len(adjacency_matrix)
    best_string = 0
    costs = collections.defaultdict(list)
    for b in range(2**number_of_qubits):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(number_of_qubits)))]
        cost = 0
        for i in range(number_of_qubits):
            for j in range(number_of_qubits):
                cost += adjacency_matrix[i,j] * x[i] * (1-x[j])

        cost = np.round(cost,5)
        x.reverse()
        costs[cost].append(x)

        if best_cost < cost:
            best_cost = cost
            best_string = x

    costs = sorted(costs.items())
    return best_cost, best_string, costs


best_cost, best_string, costs = best_cost_brute(w)
print(costs)
print(f'For the given instance the optimal cost is {best_cost} and the bitstrings corresponding to that are {costs[-1][1]}')






def get_offset(number_of_qubits, adjacency_matrix): #The offset (constant part of the Hamiltonian) for a general graph.
    offset = 0
    for i in range(number_of_qubits):
        for j in range(number_of_qubits):
            if i<j:
                offset += adjacency_matrix[i,j]/2
    return offset


def minimum_instantaneous(time): #This gives the minimum energy at a given time.
    hamil = calculate_instantaneous_hamiltonian(time)
    eigenvalues, v1 = np.linalg.eig(hamil)
    min_eig = np.min(eigenvalues) - time*offset
    return np.real(min_eig)


def calculate_spectrum(steps, plot=True):
    gs_energies = []
    first_excited_energies = []
    ener1, ener2, ener3, ener4, ener5, ener6, ener7, ener8, ener9, ener10 = [],[],[],[],[],[],[],[],[],[]
    for step in range(steps+1):
        hamiltonian = calculate_instantaneous_hamiltonian(step/steps)
        q, v = np.linalg.eigh(hamiltonian)
        q = sorted(q)
        gs_energy = q[0]
        en1, en2, en3, en4, en5, en6, en7, en8, en9, en10 = q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10]
        for k in range(len(q)):
            if np.round(q[k], 5) != np.round(gs_energy, 5):
                break
        first_excited_energy = q[k+1]

        ener1.append(en1-(step/steps)*offset)
        ener2.append(en2-(step/steps)*offset)
        ener3.append(en3-(step/steps)*offset)
        ener4.append(en4-(step/steps)*offset)
        ener5.append(en5-(step/steps)*offset)
        ener6.append(en6-(step/steps)*offset)
        ener7.append(en7-(step/steps)*offset)
        ener8.append(en8-(step/steps)*offset)
        ener9.append(en9-(step/steps)*offset)
        ener10.append(en10-(step/steps)*offset)
        gs_energies.append(gs_energy-(step/steps)*offset)
        first_excited_energies.append(first_excited_energy-(step/steps)*offset)
    
    time = [step/steps for step in range(steps+1)]
    if plot == True:
        plt.plot(time, gs_energies, label='Ground State energy')
        #plt.plot(time, first_excited_energies, label='First Excited energy')
        plt.plot(time, ener1, label='energy1')
        plt.plot(time, ener2, label='energy2')
        plt.plot(time, ener3, label='energy3')
        plt.plot(time, ener4, label='energy4')
        plt.plot(time, ener5, label='energy5')
        plt.plot(time, ener6, label='energy6')
        plt.plot(time, ener7, label='energy7')
        plt.plot(time, ener8, label='energy8')
        plt.plot(time, ener9, label='energy9')
        plt.plot(time, ener10, label='energy10')
        plt.legend(fontsize=14)
        plt.show()

    return gs_energies, first_excited_energies, ener1, ener2, ener3, ener4, ener5, ener6, ener7, ener8, ener9, ener10

gs_energies, first_excited_energies, ener1, ener2, ener3, ener4, ener5, ener6, ener7, ener8, ener9, ener10 = calculate_spectrum(steps)

print(gs_energies)
print(first_excited_energies)

#We must first calculate the linear system of equations.




#energies_aqqpqc_0_layer = [-6, -5.950000000000003, -5.900000000000002, -5.850000000000002, -5.8000000000000025, -5.7500000000000036, -5.700000000000003, -5.650000000000001, -5.600000000000002, -5.550000000000001, -5.500000000000002, -5.450000000000001, -5.400000000000001, -5.3675997603859695, -5.364692179604818, -5.384587040477438, -5.422425754662893, -5.474606432222031, -5.538409853965691, -5.611749378717045, -5.693000478625443, -5.7808807382792455, -5.874366420370951, -5.972630885535981, -6.074999988470246, -6.180918998637066, -6.289927529343797, -6.4016404219777945, -6.515733524828797, -6.631939774461214, -6.750003297851241, -6.750003297851241]
#energies_aqcpcq_1_layer = [-6, -5.950154451422959, -5.900752501028443, -5.851599417473611, -5.802810275859583, -5.756143421395298, -5.709640388324241, -5.663747298479056, -5.6187162302062434, -5.574728252892157, -5.532062018486879, -5.490699606453825, -5.4512718237156585, -5.416590678447584, -5.406928300964287, -5.4322567770114425, -5.483743933815743, -5.549963067454288, -5.637808125319982, -5.956258904387567, -6.333491366547241, -6.557759368393276, -6.794265208191857, -7.042191120246468, -7.3000746636667655, -7.56669758357127, -7.841036517600447, -8.12222509218769, -8.40952415147253, -8.702298821715617, -8.999999951495681, -8.999999951495681]
#energies_aqcpqc_2_layer = [-6, -5.950157261263083, -5.9014846312192635, -5.853497145472425, -5.80647272674259, -5.760501715222631, -5.715656725992175, -5.672441082806816, -5.630451764696526, -5.5945951732748735, -5.5570613522160475, -5.523330193226521, -5.489608225137747, -5.458743939422438, -5.457445871024463, -5.501311014037389, -5.571978192624387, -5.754371140922796, -5.933836080865668, -6.125010180756819, -6.333488907412294, -6.557177612912261, -6.7939464214127, -7.042026695042157, -7.299997078302127, -7.566666313970032, -7.84102539186039, -8.122222076713062, -8.409523724866252, -8.7022987886861, -8.99999998178523, -8.99999998178523]
#energies_aqcpqc_3_layer = []
'''