# hybrid_quantum_classical.py
import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram

def create_hamiltonian():
    """
    Create a Hamiltonian for a simple quantum system.
    
    Returns:
    - Hamiltonian as a SparsePauliOp
    """
    # Example Hamiltonian: H = 0.5 * Z0 + 0.5 * Z1 + 0.5 * Z0 Z1
    hamiltonian = SparsePauliOp.from_list([
        ('I', 1.0),  # Identity term
        ('Z', 0.5),  # Z0 term
        ('Z', 0.5),  # Z1 term
        ('ZZ', 0.5)  # Z0 Z1 term
    ])
    return hamiltonian

def create_ansatz(num_qubits):
    """
    Create a variational ansatz circuit.
    
    Parameters:
    - num_qubits: Number of qubits in the ansatz
    
    Returns:
    - QuantumCircuit: The variational ansatz circuit
    """
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.h(qubit)  # Apply Hadamard to create superposition
    return circuit

def run_vqe(hamiltonian, num_qubits):
    """
    Run the Variational Quantum Eigensolver (VQE) algorithm.
    
    Parameters:
    - hamiltonian: The Hamiltonian of the system
    - num_qubits: Number of qubits in the system
    
    Returns:
    - The minimum eigenvalue and corresponding parameters
    """
    # Create a variational ansatz
    ansatz = create_ansatz(num_qubits)

    # Choose an optimizer
    optimizer = SLSQP(maxiter=100)

    # Create a VQE instance
    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=Aer.get_backend('aer_simulator'))

    # Run VQE
    result = vqe.compute_minimum_eigenvalue(hamiltonian)

    return result.eigenvalue, result.optimal_point

def visualize_results(eigenvalue, optimal_params):
    """
    Visualize the results of the VQE simulation.
    
    Parameters:
    - eigenvalue: Minimum eigenvalue found
    - optimal_params: Optimal parameters for the ansatz
    """
    print("Minimum Eigenvalue:", eigenvalue)
    print("Optimal Parameters:", optimal_params)

if __name__ == "__main__":
    num_qubits = 2  # Number of qubits for the Hamiltonian
    hamiltonian = create_hamiltonian()  # Create the Hamiltonian

    # Run VQE
    eigenvalue, optimal_params = run_vqe(hamiltonian, num_qubits)

    # Visualize the results
    visualize_results(eigenvalue, optimal_params)
