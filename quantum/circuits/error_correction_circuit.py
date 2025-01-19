# error_correction_circuit.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def create_error_correction_circuit():
    """
    Create a quantum circuit for the 3-qubit bit-flip error correction code.
    
    Returns:
    - QuantumCircuit: The constructed error correction circuit
    """
    circuit = QuantumCircuit(5, 3)  # 5 qubits (3 data + 2 ancilla) and 3 classical bits

    # Step 1: Prepare the logical qubit |0> + |1>
    circuit.h(0)  # Prepare the first qubit in superposition

    # Step 2: Encode the logical qubit into 3 qubits
    circuit.cx(0, 1)  # |0> -> |00> and |1> -> |11>
    circuit.cx(0, 2)

    # Simulate a bit-flip error on the second qubit
    circuit.x(1)  # Introduce an error for demonstration

    # Step 3: Error correction
    # Measure the first and second qubits into ancilla qubits
    circuit.cx(1, 3)  # Check qubit 1
    circuit.cx(2, 4)  # Check qubit 2

    # Step 4: Measure ancilla qubits
    circuit.measure(3, 0)  # Measure ancilla 1
    circuit.measure(4, 1)  # Measure ancilla 2

    # Step 5: Apply correction based on measurement results
    circuit.x(1).c_if(circuit.clbits[0], 1)  # If ancilla 1 is 1, flip qubit 1
    circuit.x(2).c_if(circuit.clbits[1], 1)  # If ancilla 2 is 1, flip qubit 2

    # Measure the corrected logical qubit
    circuit.measure(1, 2)  # Measure the first data qubit

    return circuit

def run_error_correction_simulation():
    """
    Run the quantum error correction simulation and return the results.
    
    Returns:
    - counts: Measurement results
    """
    # Create the error correction circuit
    circuit = create_error_correction_circuit()

    # Use the Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1024)
    result = job.result()

    # Get measurement counts
    counts = result.get_counts(circuit)

    return counts

def visualize_results(counts):
    """
    Visualize the results of the error correction simulation.
    
    Parameters:
    - counts: Measurement results
    """
    print("Counts:", counts)
    plot_histogram(counts).show()

if __name__ == "__main__":
    # Run the error correction simulation
    counts = run_error_correction_simulation()

    # Visualize the results
    visualize_results(counts)
