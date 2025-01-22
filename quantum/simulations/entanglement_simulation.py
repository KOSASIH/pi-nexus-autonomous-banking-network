# entanglement_simulation.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector

def create_entangled_state():
    """
    Create a quantum circuit that generates a Bell state (entangled state).
    
    Returns:
    - QuantumCircuit: The constructed quantum circuit
    """
    circuit = QuantumCircuit(2, 2)  # 2 qubits and 2 classical bits

    # Apply Hadamard gate to the first qubit
    circuit.h(0)

    # Apply CNOT gate to create entanglement
    circuit.cx(0, 1)

    # Measure the qubits
    circuit.measure(range(2), range(2))

    return circuit

def run_entanglement_simulation():
    """
    Run the entanglement simulation and return the results.
    
    Returns:
    - Counts of the measurement results
    - Statevector of the quantum system
    """
    # Create the entangled state circuit
    circuit = create_entangled_state()

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

def visualize_results(counts, statevector):
    """
    Visualize the results of the entanglement simulation.
    
    Parameters:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    print("Counts:", counts)
    plot_histogram(counts).show()
    plot_bloch_multivector(statevector).show()

if __name__ == "__main__":
    # Run the entanglement simulation
    counts, statevector = run_entanglement_simulation()

    # Visualize the results
    visualize_results(counts, statevector)
