# quantum_simulation_algorithms.py
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

class QuantumPhaseEstimation:
    def __init__(self, unitary, num_qubits):
        """
        Initialize the Quantum Phase Estimation algorithm.

        Parameters:
        - unitary: The unitary operator (as a quantum circuit) whose eigenvalue is to be estimated.
        - num_qubits: Number of qubits for the phase estimation.
        """
        self.unitary = unitary
        self.num_qubits = num_qubits

    def create_circuit(self):
        """
        Create the quantum circuit for the Quantum Phase Estimation algorithm.

        Returns:
        - QuantumCircuit: The constructed QPE circuit.
        """
        # Create a quantum circuit with additional qubits for the phase estimation
        circuit = QuantumCircuit(self.num_qubits + 1, self.num_qubits)

        # Initialize the counting qubits to |+>
        for i in range(self.num_qubits):
            circuit.h(i)

        # Apply controlled unitary operations
        for i in range(self.num_qubits):
            circuit.append(self.unitary.control(), [i] + [self.num_qubits])

        # Apply inverse QFT
        circuit.append(self.inverse_qft(self.num_qubits), range(self.num_qubits))

        # Measure the counting qubits
        circuit.measure(range(self.num_qubits), range(self.num_qubits))

        return circuit

    def inverse_qft(self, n):
        """
        Create the inverse Quantum Fourier Transform circuit.

        Parameters:
        - n: Number of qubits.

        Returns:
        - QuantumCircuit: The inverse QFT circuit.
        """
        circuit = QuantumCircuit(n)
        for j in range(n):
            circuit.h(j)
            for k in range(j):
                circuit.cp(-np.pi / (2 ** (j - k)), k, j)
        return circuit

    def estimate_phase(self):
        """
        Estimate the phase using the Quantum Phase Estimation algorithm.

        Returns:
        - result: The measurement result of the phase estimation.
        """
        # Create the QPE circuit
        circuit = self.create_circuit()

        # Execute the circuit
        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(circuit, backend)
        qobj = assemble(transpiled_circuit)
        result = execute(qobj, backend, shots=1024).result()

        return result.get_counts()

if __name__ == "__main__":
    # Example usage
    # Define a unitary operator (e.g., a rotation operator)
    theta = np.pi / 4  # Example angle
    unitary = QuantumCircuit(1)
    unitary.rz(theta, 0)  # Apply a rotation around the Z-axis

    # Initialize the Quantum Phase Estimation algorithm
    qpe = QuantumPhaseEstimation(unitary, num_qubits=3)

    # Estimate the phase
    counts = qpe.estimate_phase()
    print("Measurement Results:", counts)

    # Plot the results
    plot_histogram(counts)
