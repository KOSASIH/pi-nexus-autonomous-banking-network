import numpy as np
from qiskit import QuantumCircuit, execute, Aer

class QuantumComputing:
    def __init__(self, backend):
        self.backend = backend

    def create_circuit(self, num_qubits):
        circuit = QuantumCircuit(num_qubits)
        return circuit

    def add_hadamard_gate(self, circuit, qubit):
        circuit.h(qubit)
        return circuit

    def add_pauli_x_gate(self, circuit, qubit):
        circuit.x(qubit)
        return circuit

    def add_pauli_y_gate(self, circuit, qubit):
        circuit.y(qubit)
        return circuit

    def add_pauli_z_gate(self, circuit, qubit):
        circuit.z(qubit)
        return circuit

    def measure_circuit(self, circuit):
        job = execute(circuit, self.backend)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

    def run_quantum_algorithm(self, circuit):
        # Run a quantum algorithm (e.g. Shor's algorithm)
        # For simplicity, we'll just run a Hadamard gate on each qubit
        for qubit in range(circuit.num_qubits):
            circuit = self.add_hadamard_gate(circuit, qubit)
        return circuit

# Example usage:
backend = Aer.get_backend('qasm_simulator')
quantum_computing = QuantumComputing(backend)

num_qubits = 5
circuit = quantum_computing.create_circuit(num_qubits)

circuit = quantum_computing.add_hadamard_gate(circuit, 0)
circuit = quantum_computing.add_pauli_x_gate(circuit, 1)
circuit = quantum_computing.add_pauli_y_gate(circuit, 2)
circuit = quantum_computing.add_pauli_z_gate(circuit, 3)

counts = quantum_computing.measure_circuit(circuit)
print(counts)

circuit = quantum_computing.run_quantum_algorithm(circuit)
counts = quantum_computing.measure_circuit(circuit)
print(counts)
