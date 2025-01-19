# quantum_state_visualization.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_state_qsphere, plot_histogram
from qiskit.quantum_info import Statevector

def create_sample_circuit():
    """
    Create a sample quantum circuit to generate a quantum state.
    
    Returns:
    - QuantumCircuit: The constructed quantum circuit
    """
    circuit = QuantumCircuit(1)  # 1 qubit

    # Prepare a superposition state
    circuit.h(0)  # Apply Hadamard gate to create |+> state

    return circuit

def run_circuit_and_get_statevector(circuit):
    """
    Run the quantum circuit and return the state vector.
    
    Parameters:
    - circuit: QuantumCircuit object
    
    Returns:
    - Statevector: The state vector of the quantum system
    """
    # Use the Aer's statevector simulator
    simulator = Aer.get_backend('statevector_simulator')

    # Execute the circuit
    job = execute(circuit, simulator)
    result = job.result()

    # Get the state vector
    statevector = result.get_statevector(circuit)

    return statevector

def visualize_state(statevector):
    """
    Visualize the quantum state using Bloch sphere and Q-sphere representations.
    
    Parameters:
    - statevector: State vector of the quantum system
    """
    print("State Vector:", statevector)

    # Plot Bloch sphere representation
    plot_bloch_multivector(statevector).show()

    # Plot Q-sphere representation
    plot_state_qsphere(statevector).show()

if __name__ == "__main__":
    # Create a sample quantum circuit
    circuit = create_sample_circuit()

    # Run the circuit and get the state vector
    statevector = run_circuit_and_get_statevector(circuit)

    # Visualize the quantum state
    visualize_state(statevector)
