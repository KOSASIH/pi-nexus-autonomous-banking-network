# quantum_state.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector

class QuantumState:
    def __init__(self, num_qubits):
        """
        Initialize a quantum state with a specified number of qubits.
        
        Parameters:
        - num_qubits: Number of qubits in the quantum state
        """
        self.num_qubits = num_qubits
        self.statevector = Statevector.from_dict({(0,)*num_qubits: 1})  # Start in |0...0> state
        self.circuit = QuantumCircuit(num_qubits)

    def apply_hadamard(self, qubit):
        """
        Apply a Hadamard gate to a specified qubit.
        
        Parameters:
        - qubit: The index of the qubit to which the Hadamard gate will be applied
        """
        self.circuit.h(qubit)
        self.update_state()

    def apply_cnot(self, control, target):
        """
        Apply a CNOT gate with a specified control and target qubit.
        
        Parameters:
        - control: The index of the control qubit
        - target: The index of the target qubit
        """
        self.circuit.cx(control, target)
        self.update_state()

    def update_state(self):
        """
        Update the state vector based on the current circuit.
        """
        self.statevector = Statevector.from_dict(self.circuit.save_statevector().data)

    def measure(self):
        """
        Measure the quantum state and return the measurement results.
        
        Returns:
        - counts: Measurement results
        """
        self.circuit.measure_all()
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(self.circuit, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts(self.circuit)
        return counts

    def visualize(self):
        """
        Visualize the current quantum state on the Bloch sphere.
        """
        plot_bloch_multivector(self.statevector).show()

    def prepare_state(self, state_vector):
        """
        Prepare a quantum state from a given state vector.
        
        Parameters:
        - state_vector: List or array representing the quantum state
        """
        self.circuit.initialize(state_vector, range(self.num_qubits))
        self.update_state()

    def __str__(self):
        """
        String representation of the quantum state.
        """
        return f"QuantumState(num_qubits={self.num_qubits}, statevector={self.statevector})"

if __name__ == "__main__":
    # Example usage of the QuantumState class
    num_qubits = 2
    q_state = QuantumState(num_qubits)

    # Apply Hadamard to the first qubit
    q_state.apply_hadamard(0)

    # Apply CNOT with qubit 0 as control and qubit 1 as target
    q_state.apply_cnot(0, 1)

    # Measure the state
    counts = q_state.measure()
    print("Measurement Results:", counts)

    # Visualize the quantum state
    q_state.visualize()
