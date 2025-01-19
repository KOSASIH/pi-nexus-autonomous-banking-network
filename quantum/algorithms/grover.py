# grover.py
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

def oracle(circuit, n, target):
    """
    Oracle for Grover's algorithm.
    Marks the target state by flipping its amplitude.
    
    Parameters:
    - circuit: QuantumCircuit object
    - n: Number of qubits
    - target: Target state as a binary string
    """
    # Convert target state to integer
    target_int = int(target, 2)
    
    # Apply X gates to the target state
    for qubit in range(n):
        if (target_int >> qubit) & 1 == 0:
            circuit.x(qubit)
    
    # Apply controlled-Z gate
    circuit.h(n-1)  # Apply Hadamard to the last qubit
    circuit.mct(list(range(n-1)), n-1)  # Multi-controlled Toffoli
    circuit.h(n-1)  # Apply Hadamard to the last qubit again
    
    # Apply X gates to revert the target state
    for qubit in range(n):
        if (target_int >> qubit) & 1 == 0:
            circuit.x(qubit)

def grover_search(n, target, iterations):
    """
    Grover's search algorithm.
    
    Parameters:
    - n: Number of qubits
    - target: Target state as a binary string
    - iterations: Number of iterations of Grover's algorithm
    """
    # Create a quantum circuit with n qubits and n classical bits
    circuit = QuantumCircuit(n, n)
    
    # Initialize qubits to |+>
    circuit.h(range(n))
    
    # Apply Grover's iterations
    for _ in range(iterations):
        oracle(circuit, n, target)  # Apply the oracle
        circuit.h(range(n))          # Apply Hadamard gates
        circuit.x(range(n))          # Apply X gates
        circuit.h(n-1)               # Apply Hadamard to the last qubit
        circuit.mct(list(range(n-1)), n-1)  # Multi-controlled Toffoli
        circuit.h(n-1)               # Apply Hadamard to the last qubit again
        circuit.x(range(n))          # Apply X gates
        circuit.h(range(n))          # Apply Hadamard gates again
    
    # Measure the qubits
    circuit.measure(range(n), range(n))
    
    return circuit

def run_grover(n, target):
    """
    Run Grover's algorithm and return the results.
    
    Parameters:
    - n: Number of qubits
    - target: Target state as a binary string
    """
    # Calculate the number of iterations
    iterations = int(np.pi / 4 * np.sqrt(2**n))
    
    # Create and run the Grover circuit
    circuit = grover_search(n, target, iterations)
    
    # Use the Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1024)
    
    # Grab results from the job
    result = job.result()
    
    # Returns counts
    counts = result.get_counts(circuit)
    
    return counts

if __name__ == "__main__":
    n = 3  # Number of qubits
    target = '101'  # Target state to search for
    counts = run_grover(n, target)
    
    # Print the results
    print("Counts:", counts)
    
    # Plot the results
    plot_histogram(counts).show()
