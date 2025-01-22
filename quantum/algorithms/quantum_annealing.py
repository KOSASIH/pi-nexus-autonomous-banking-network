# quantum_annealing.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp

def create_ising_hamiltonian(J, h, num_qubits):
    """
    Create the Ising Hamiltonian for a given number of qubits.
    
    Parameters:
    - J: Coupling strength between qubits
    - h: External magnetic field
    - num_qubits: Number of qubits
    
    Returns:
    - Hamiltonian as a SparsePauliOp
    """
    # Initialize the Hamiltonian
    hamiltonian = SparsePauliOp.from_list([])
    
    # Add interaction terms (J * Z_i * Z_j)
    for i in range(num_qubits):
        hamiltonian += SparsePauliOp.from_list([('Z' * num_qubits, J)])  # Z_i Z_j terms

    # Add external field terms (h * Z_i)
    for i in range(num_qubits):
        hamiltonian += SparsePauliOp.from_list([('Z' * i + 'I' + 'Z' * (num_qubits - i - 1), h)])  # Z_i terms

    return hamiltonian

def quantum_annealing(hamiltonian):
    """
    Simulate quantum annealing to find the ground state of the Hamiltonian.
    
    Parameters:
    - hamiltonian: The Hamiltonian of the system
    
    Returns:
    - The minimum eigenvalue and corresponding state
    """
    # Use the NumPyMinimumEigensolver to find the ground state
    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(hamiltonian)

    return result.eigenvalue, result.eigenstate

if __name__ == "__main__":
    num_qubits = 3  # Number of qubits
    J = -1.0  # Coupling strength
    h = 0.5   # External magnetic field

    # Create the Ising Hamiltonian
    hamiltonian = create_ising_hamiltonian(J, h, num_qubits)

    # Run quantum annealing
    eigenvalue, ground_state = quantum_annealing(hamiltonian)

    # Print the results
    print("Minimum Eigenvalue (Ground State Energy):", eigenvalue)
    print("Ground State:", ground_state)
