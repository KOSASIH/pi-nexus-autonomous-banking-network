# quantum_state_preparation.py
from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_multivector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np

def prepare_ghz_state(n):
    """
    Prepare a GHZ state for n qubits.

    Parameters:
    - n: Number of qubits.

    Returns:
    - QuantumCircuit: The circuit that prepares the GHZ state.
    """
    circuit = QuantumCircuit(n)
    circuit.h(0)  # Apply Hadamard to the first qubit
    for i in range(1, n):
        circuit.cx(0, i)  # Apply CNOT gates to create entanglement
    return circuit

def prepare_w_state(n):
    """
    Prepare a W state for n qubits.

    Parameters:
    - n: Number of qubits.

    Returns:
    - QuantumCircuit: The circuit that prepares the W state.
    """
    circuit = QuantumCircuit(n)
    for i in range(n - 1):
        circuit.h(i)  # Apply Hadamard to the first n-1 qubits
        circuit.cx(i, n - 1)  # Apply CNOT to the last qubit
    circuit.h(n - 1)  # Apply Hadamard to the last qubit
    return circuit

def visualize_state(circuit):
    """
    Visualize the quantum state prepared by the circuit.

    Parameters:
    - circuit: The quantum circuit that prepares the state.
    """
    # Simulate the circuit to get the statevector
    simulator = AerSimulator()
    circuit.save_statevector()
    result = simulator.run(circuit).result()
    statevector = result.get_statevector(circuit)

    # Plot the Bloch vector representation of the state
    plot_bloch_multivector(statevector)
    print("Statevector:", statevector)

if __name__ == "__main__":
    # Example usage for GHZ state
    n_qubits_ghz = 3
    ghz_circuit = prepare_ghz_state(n_qubits_ghz)
    print("GHZ State Circuit:")
    print(ghz_circuit.draw())

    # Visualize the GHZ state
    visualize_state(ghz_circuit)

    # Example usage for W state
    n_qubits_w = 3
    w_circuit = prepare_w_state(n_qubits_w)
    print("\nW State Circuit:")
    print(w_circuit.draw())

    # Visualize the W state
    visualize_state(w_circuit)
