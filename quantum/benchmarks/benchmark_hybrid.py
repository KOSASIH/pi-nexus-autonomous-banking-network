# benchmark_hybrid.py
import numpy as np
import time
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Sampler

def create_qaoa_circuit(p):
    """
    Create a QAOA circuit for a simple optimization problem.
    
    Parameters:
    - p: Number of layers in the QAOA circuit
    
    Returns:
    - QuantumCircuit: The constructed QAOA circuit
    """
    circuit = TwoLocal(2, rotation_blocks='ry', entanglement='cz', reps=p)
    return circuit

def run_hybrid_benchmark(p, shots=1024):
    """
    Run a benchmark for the hybrid quantum-classical algorithm (QAOA).
    
    Parameters:
    - p: Number of layers in the QAOA circuit
    - shots: Number of shots for the simulation
    
    Returns:
    - execution_time: Time taken to execute the algorithm
    - solution: Estimated solution from the QAOA
    """
    # Create the QAOA circuit
    circuit = create_qaoa_circuit(p)
    
    # Set up the QAOA algorithm
    optimizer = SLSQP(maxiter=100)
    qaoa = QAOA(ansatz=circuit, optimizer=optimizer, quantum_instance=Aer.get_backend('aer_simulator'))
    
    # Run the QAOA algorithm
    start_time = time.time()
    result = qaoa.compute_minimum_eigenvalue()
    execution_time = time.time() - start_time
    
    # Extract the solution
    solution = result.eigenvalue.real
    return execution_time, solution

def benchmark_hybrid(max_layers, shots=1024):
    """
    Benchmark the hybrid quantum-classical algorithm for different numbers of layers.
    
    Parameters:
    - max_layers: Maximum number of layers to test
    - shots: Number of shots for each simulation
    
    Returns:
    - results: List of tuples containing (p, execution_time, solution)
    """
    results = []
    
    for p in range(1, max_layers + 1):
        execution_time, solution = run_hybrid_benchmark(p, shots)
        results.append((p, execution_time, solution))
        print(f"Benchmarking: Layers={p}, Execution Time={execution_time:.4f}s, Solution={solution:.4f}")
    
    return results

def plot_benchmark_results(results):
    """
    Plot the benchmark results for the hybrid quantum-classical algorithm.
    
    Parameters:
    - results: List of tuples containing (p, execution_time, solution)
    """
    layers = [r[0] for r in results]
    execution_times = [r[1] for r in results]
    solutions = [r[2] for r in results]
    
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Number of Layers (p)')
    ax1.set_ylabel('Execution Time (s)', color=color)
    ax1.plot(layers, execution_times, color=color, label='Execution Time')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Solution Value', color=color)
    ax2.plot(layers, solutions, color=color, label='Solution Value')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('Hybrid Quantum-Classical Algorithm Benchmark Results')
    plt.show()

if __name__ == "__main__":
    max_layers = 5  # Set the maximum number of layers to benchmark
    shots = 1024     # Set the number of shots for each simulation
    results = benchmark_hybrid(max_layers, shots)
    plot_benchmark_results(results)
