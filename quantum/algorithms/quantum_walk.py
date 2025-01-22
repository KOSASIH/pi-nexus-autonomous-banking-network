# quantum_walk.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def create_quantum_walk_circuit(steps):
    """
    Create a quantum circuit for a one-dimensional quantum walk.
    
    Parameters:
    - steps: Number of steps in the quantum walk
    
    Returns:
    - QuantumCircuit: The quantum circuit for the quantum walk
    """
    # Create a quantum circuit with 2 qubits: one for position and one for the coin
    circuit = QuantumCircuit(2)

    # Initialize the coin state to |+>
    circuit.h(1)  # Apply Hadamard to the coin qubit

    # Apply the quantum walk for the specified number of steps
    for _ in range(steps):
        # Apply the coin operation
        circuit.ry(np.pi / 2, 1)  # Rotate the coin qubit

        # Apply the conditional shift operation
        circuit.cx(1, 0)  # If coin is |1>, move right
        circuit.x(0)      # Flip the position qubit
        circuit.cx(1, 0)  # If coin is |0>, move left
        circuit.x(0)      # Revert the position qubit

    return circuit

def run_quantum_walk(steps):
    """
    Run the quantum walk and return the results.
    
    Parameters:
    - steps: Number of steps in the quantum walk
    
    Returns:
    - Counts of the measurement results
    """
    # Create the quantum walk circuit
    circuit = create_quantum_walk_circuit(steps)

    # Measure the position qubit
    circuit.measure_all()

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
    steps = 3  # Number of steps in the quantum walk
    counts = run_quantum_walk(steps)

    # Print the results
    print("Counts:", counts)

    # Plot the results
    plot_histogram(counts).show()
