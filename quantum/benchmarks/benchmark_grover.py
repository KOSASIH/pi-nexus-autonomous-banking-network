# benchmark_grover.py
import numpy as np
import time
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def create_grover_circuit(num_qubits, marked_element):
    """
    Create a Grover's algorithm circuit for a given number of qubits and marked element.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - marked_element: The index of the marked element to search for
    
    Returns:
    - QuantumCircuit: The constructed Grover's circuit
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Initialize qubits in superposition
    circuit.h(range(num_qubits))
    
    # Grover's iterations
    num_iterations = int(np.pi / 4 * np.sqrt(2**num_qubits))
    for _ in range(num_iterations):
        # Oracle: Flip the sign of the marked element
        circuit.x(marked_element)
        circuit.h(marked_element)
        circuit.z(marked_element)
        circuit.h(marked_element)
        circuit.x(marked_element)
        
        # Diffusion operator
        circuit.h(range(num_qubits))
        circuit.x(range(num_qubits))
        circuit.h(num_qubits - 1)
        circuit.cx(range(num_qubits - 1), num_qubits - 1)
        circuit.h(num_qubits - 1)
        circuit.x(range(num_qubits))
        circuit.h(range(num_qubits))
    
    # Measure the qubits
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit

def run_grover_benchmark(num_qubits, marked_element, shots=1024):
    """
    Run a benchmark for Grover's algorithm and return execution time and success probability.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - marked_element: The index of the marked element to search for
    - shots: Number of shots for the simulation
    
    Returns:
    - execution_time: Time taken to execute the circuit
    - success_probability: Probability of measuring the marked element
    """
    circuit = create_grover_circuit(num_qubits, marked_element)
    
    # Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    start_time = time.time()
    result = execute(circuit, backend, shots=shots).result()
    execution_time = time.time() - start_time
    
    # Get measurement counts
    counts = result.get_counts(circuit)
    success_count = counts.get(bin(marked_element)[2:].zfill(num_qubits), 0)
    success_probability = success_count / shots
    
    return execution_time, success_probability

def benchmark_grover(max_qubits, shots=1024):
    """
    Benchmark Grover's algorithm for different numbers of qubits and marked elements.
    
    Parameters:
    - max_qubits: Maximum number of qubits to test
    - shots: Number of shots for each simulation
    
    Returns:
    - results: List of tuples containing (num_qubits, marked_element, execution_time, success_probability)
    """
    results = []
    
    for num_qubits in range(2, max_qubits + 1):
        for marked_element in range(2**num_qubits):
            execution_time, success_probability = run_grover_benchmark(num_qubits, marked_element, shots)
            results.append((num_qubits, marked_element, execution_time, success_probability))
            print(f"Benchmarking: Qubits={num_qubits}, Marked Element={marked_element}, "
                  f"Execution Time={execution_time:.4f}s, Success Probability={success_probability:.4f}")
    
    return results

def plot_benchmark_results(results):
    """
    Plot the benchmark results for Grover's algorithm.
    
    Parameters:
    - results: List of tuples containing (num_qubits, marked_element, execution_time, success_probability)
    """
    num_qubits = [r[0] for r in results]
    execution_times = [r[2] for r in results]
    success_probabilities = [r[3] for r in results]
    
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Execution Time (s)', color=color)
    ax1.plot(num_qubits, execution_times, color=color, label='Execution Time')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Success Probability', color=color)
    ax2.plot(num_qubits, success_probabilities, color=color, label='Success Probability')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('Grover\'s Algorithm Benchmark Results')
    plt.show()

if __name__ == "__main__":
    max_qubits = 5  # Set the maximum number of qubits to benchmark
    shots = 1024    # Set the number of shots for each simulation
    results = benchmark_grover(max_qubits, shots)
    plot_benchmark_results(results)
