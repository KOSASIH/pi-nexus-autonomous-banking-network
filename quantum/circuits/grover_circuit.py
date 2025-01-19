# grover_circuit.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

def create_grover_circuit(num_qubits, marked_element):
    """
    Create a quantum circuit for Grover's algorithm.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - marked_element: The index of the marked element to search for (0 to 2^num_qubits - 1)
    
    Returns:
    - QuantumCircuit: The constructed Grover's algorithm circuit
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)

    # Initialize qubits in superposition
    circuit.h(range(num_qubits))

    # Grover's iterations
    num_iterations = int(np.pi / 4 * np.sqrt(2**num_qubits))  # Optimal number of iterations
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

def run_grover_simulation(num_qubits, marked_element):
    """
    Run the Grover's algorithm simulation and return the results.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - marked_element: The index of the marked element to search for
    
    Returns:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    # Create the Grover circuit
    circuit = create_grover_circuit(num_qubits, marked_element)

    # Use the Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1024)
    result = job.result()

    # Get measurement counts
    counts = result.get_counts(circuit)

    # Get the state vector for visualization
    statevector = Statevector.from_dict(counts)

    return counts, statevector

def visualize_results(counts):
    """
    Visualize the results of the Grover's algorithm simulation.
    
    Parameters:
    - counts: Measurement results
    """
    print("Counts:", counts)
    plot_histogram(counts).show()

if __name__ == "__main__":
    num_qubits = 3  # Number of qubits (for 8 possible elements)
    marked_element = 5  # The index of the marked element (e.g., 5)

    # Run the Grover simulation
    counts, statevector = run_grover_simulation(num_qubits, marked_element)

    # Visualize the results
    visualize_results(counts)
