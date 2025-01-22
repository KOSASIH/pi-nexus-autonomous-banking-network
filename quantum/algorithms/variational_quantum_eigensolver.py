# variational_quantum_eigensolver.py
import numpy as np
from qiskit import Aer, QuantumCircuit, transpile, assemble
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit.quantum_info import Pauli
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.quantum_info import PauliExpectation
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

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

def vqe(hamiltonian, num_qubits):
    """
    Run the Variational Quantum Eigensolver (VQE) algorithm.
    
    Parameters:
    - hamiltonian: The Hamiltonian of the system
    - num_qubits: Number of qubits in the system
    
    Returns:
    - The minimum eigenvalue and corresponding parameters
    """
    # Create a variational ansatz
    ansatz = TwoLocal(num_qubits, rotation_blocks='ry', entanglement='cz', reps=2)

    # Choose an optimizer
    optimizer = SLSQP(maxiter=100)

    # Create a VQE instance
    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=Aer.get_backend('aer_simulator'))

    # Run VQE
    result = vqe.compute_minimum_eigenvalue(hamiltonian)

    return result.eigenvalue, result.optimal_point

if __name__ == "__main__":
    num_qubits = 2  # Number of qubits for the Hamiltonian
    hamiltonian = create_hamiltonian()  # Create the Hamiltonian
    eigenvalue, optimal_params = vqe(hamiltonian, num_qubits)  # Run VQE

    # Print the results
    print("Minimum Eigenvalue:", eigenvalue)
    print("Optimal Parameters:", optimal_params)
