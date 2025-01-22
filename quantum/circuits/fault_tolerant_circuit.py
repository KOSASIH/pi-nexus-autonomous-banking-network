# fault_tolerant_circuit.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def create_fault_tolerant_circuit():
    """
    Create a quantum circuit for the Steane code (7-qubit code).
    
    Returns:
    - QuantumCircuit: The constructed fault-tolerant circuit
    """
    circuit = QuantumCircuit(7, 7)  # 7 qubits for the Steane code and 7 classical bits for measurement

    # Step 1: Encode the logical qubit |0> into the Steane code
    circuit.h(0)  # Prepare the logical |+> state
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.cx(0, 4)
    circuit.cx(0, 5)
    circuit.cx(0, 6)

    # Step 2: Simulate a bit-flip error on one of the qubits (for demonstration)
    circuit.x(1)  # Introduce an error on qubit 1

    # Step 3: Error detection
    # Measure the parity of the first three qubits
    circuit.cx(1, 3)
    circuit.cx(2, 3)
    circuit.measure(3, 0)  # Measure parity into classical bit 0

    # Measure the parity of the last four qubits
    circuit.cx(4, 6)
    circuit.cx(5, 6)
    circuit.measure(6, 1)  # Measure parity into classical bit 1

    # Step 4: Apply correction based on measurement results
    circuit.x(1).c_if(circuit.clbits[0], 1)  # If parity is 1, flip qubit 1
    circuit.x(2).c_if(circuit.clbits[1], 1)  # If parity is 1, flip qubit 2

    # Step 5: Measure the logical qubit
    circuit.measure(range(7), range(7))  # Measure all qubits

    return circuit

def run_fault_tolerant_simulation():
    """
    Run the fault-tolerant quantum computing simulation and return the results.
    
    Returns:
    - counts: Measurement results
    """
    # Create the fault-tolerant circuit
    circuit = create_fault_tolerant_circuit()

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
    Visualize the results of the fault-tolerant simulation.
    
    Parameters:
    - counts: Measurement results
    """
    print("Counts:", counts)
    plot_histogram(counts).show()

if __name__ == "__main__":
    # Run the fault-tolerant simulation
    counts = run_fault_tolerant_simulation()

    # Visualize the results
    visualize_results(counts)
