# benchmark_vqe.py
import numpy as np
import time
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.quantum_info import SparsePauliOp

def create_hamiltonian():
    """
    Create a Hamiltonian for the H2 molecule.
    
    Returns:
    - Hamiltonian: A SparsePauliOp representing the Hamiltonian
    """
    # Example Hamiltonian for H2 molecule
    hamiltonian = (-1/2) * (SparsePauliOp.from_list([('ZZ', -1), ('XX', -1)]))
    return hamiltonian

def run_vqe_benchmark(hamiltonian, num_qubits, shots=1024):
    """
    Run a benchmark for VQE to estimate the ground state energy of a given Hamiltonian.
    
    Parameters:
    - hamiltonian: The Hamiltonian to minimize
    - num_qubits: Number of qubits in the circuit
    - shots: Number of shots for the simulation
    
    Returns:
    - execution_time: Time taken to execute the algorithm
    - ground_state_energy: Estimated ground state energy
    """
    ansatz = TwoLocal(num_qubits, rotation_blocks='ry', entanglement='cz', reps=2)
    optimizer = SLSQP(maxiter=100)
    
    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=Aer.get_backend('aer_simulator'))
    
    start_time = time.time()
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    execution_time = time.time() - start_time
    
    ground_state_energy = result.eigenvalue.real
    return execution_time, ground_state_energy

def benchmark_vqe(max_qubits, shots=1024):
    """
    Benchmark VQE for different numbers of qubits.
    
    Parameters:
    - max_qubits: Maximum number of qubits to test
    - shots: Number of shots for each simulation
    
    Returns:
    - results: List of tuples containing (num_qubits, execution_time, ground_state_energy)
    """
    results = []
    hamiltonian = create_hamiltonian()
    
    for num_qubits in range(2, max_qubits + 1):
        execution_time, ground_state_energy = run_vqe_benchmark(hamiltonian, num_qubits, shots)
        results.append((num_qubits, execution_time, ground_state_energy))
        print(f"Benchmarking: Qubits={num_qubits}, Execution Time={execution_time:.4f}s, "
              f"Ground State Energy={ground_state_energy:.4f}")
    
    return results

def plot_benchmark_results(results):
    """
    Plot the benchmark results for VQE.
    
    Parameters:
    - results: List of tuples containing (num_qubits, execution_time, ground_state_energy)
    """
    num_qubits = [r[0] for r in results]
    execution_times = [r[1] for r in results]
    ground_state_energies = [r[2] for r in results]
    
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Execution Time (s)', color=color)
    ax1.plot(num_qubits, execution_times, color=color, label='Execution Time')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Ground State Energy', color=color)
    ax2.plot(num_qubits, ground_state_energies, color=color, label='Ground State Energy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('VQE Benchmark Results')
    plt.show()

if __name__ == "__main__":
    max_qubits = 5  # Set the maximum number of qubits to benchmark
    shots = 1024    # Set the number of shots for each simulation
    results = benchmark_vqe(max_qubits, shots)
    plot_benchmark_results(results)
