# benchmark_shor.py
import numpy as np
import time
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.algorithms import Shor
from qiskit.utils import QuantumInstance

def run_shor_benchmark(n, shots=1024):
    """
    Run a benchmark for Shor's algorithm to factor a given integer n.
    
    Parameters:
    - n: Integer to factor
    - shots: Number of shots for the simulation
    
    Returns:
    - execution_time: Time taken to execute the algorithm
    - factors: Factors of the integer n found by Shor's algorithm
    """
    shor = Shor()
    quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator'), shots=shots)
    
    start_time = time.time()
    factors = shor.factor(n, quantum_instance=quantum_instance)
    execution_time = time.time() - start_time
    
    return execution_time, factors

def benchmark_shor(max_n, shots=1024):
    """
    Benchmark Shor's algorithm for different integers to factor.
    
    Parameters:
    - max_n: Maximum integer to test
    - shots: Number of shots for each simulation
    
    Returns:
    - results: List of tuples containing (n, execution_time, factors)
    """
    results = []
    
    for n in range(2, max_n + 1):
        execution_time, factors = run_shor_benchmark(n, shots)
        results.append((n, execution_time, factors))
        print(f"Benchmarking: n={n}, Execution Time={execution_time:.4f}s, Factors={factors}")
    
    return results

def plot_benchmark_results(results):
    """
    Plot the benchmark results for Shor's algorithm.
    
    Parameters:
    - results: List of tuples containing (n, execution_time, factors)
    """
    n_values = [r[0] for r in results]
    execution_times = [r[1] for r in results]
    
    plt.plot(n_values, execution_times, marker='o')
    plt.title('Shor\'s Algorithm Benchmark Results')
    plt.xlabel('Integer to Factor (n)')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    max_n = 15  # Set the maximum integer to benchmark
    shots = 1024  # Set the number of shots for each simulation
    results = benchmark_shor(max_n, shots)
    plot_benchmark_results(results)
