# quantum_fourier_transform.py
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

def qft(circuit, n):
    """
    Apply the Quantum Fourier Transform to the first n qubits in the circuit.
    
    Parameters:
    - circuit: QuantumCircuit object
    - n: Number of qubits to apply QFT to
    """
    for j in range(n):
        # Apply Hadamard gate to the j-th qubit
        circuit.h(j)
        
        # Apply controlled phase rotations
        for k in range(j + 1, n):
            circuit.cp(np.pi / 2**(k - j), k, j)  # Controlled phase rotation

    # Swap the qubits to reverse the order
    for j in range(n // 2):
        circuit.swap(j, n - j - 1)

def run_qft(n):
    """
    Create and run a quantum circuit that applies the Quantum Fourier Transform.
    
    Parameters:
    - n: Number of qubits to apply QFT to
    
    Returns:
    - Counts of the measurement results
    """
    # Create a quantum circuit with n qubits
    circuit = QuantumCircuit(n, n)

    # Apply QFT
    qft(circuit, n)

    # Measure the qubits
    circuit.measure(range(n), range(n))

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
    n = 3  # Number of qubits for the QFT
    counts = run_qft(n)

    # Print the results
    print("Counts:", counts)

    # Plot the results
    plot_histogram(counts).show()
