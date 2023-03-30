# AQC-PQC
Qiskit implementation of the paper "Adiabatic quantum computing with parameterized quantum circuits" (arXiv:2206.04373)
Authors: Ioannis Kolotouros, Ioannis Petrongonas, Milos Prokop, Petros Wallden

## Description
AQC-PQC is a hybrid quantum/classical algorithm that can run on near-term quantum devices. The algorithm starts by employing a parameterized quantum circuit and initializing
the parameters so that they produce the ground state of a known Hamiltonian. Then, at each step, a small perturbation is added in the Hamiltonian and and a (constained) 
system of linear equations is formed by calculating a series of observables on the unperturbed system. The solution of the (constrained) linear system is the \emph{shift vector} that
will translate the system to the ground state (or an approximation to it) of the perturbed system. By iteratively introducing perturbations, we can hop between ground states
so that at the end of the algorithm, the quantum mechanical system will reach the ground state of the target Hamiltonian.

## Installation
pip install -r requirements.txt

## Instructions

quantum_circuit.py -- This file constructs the parameterized quantum circuit. The single qubit gates, entanglement gates, layers and connectivity should change from main.py.

hamiltonian.py -- This file constructs the 2^n x 2^n (n=Number of Qubits) representation of the Hamiltonian. The implementation includes the Hamiltonian for the MaxCut and Number Partitioning problems (classical combinatorial optimization problems) and the Transverse-Field Ising Chain Model (quantum spin-configuration problem).

aqc_qpc.py -- This file contains the implementation of the AQC-PQC algorithm. 

main.py -- This file executes the AQC-PQC algorithm. Circuit architecture, type of problem, instance, #steps should all change from this file.




