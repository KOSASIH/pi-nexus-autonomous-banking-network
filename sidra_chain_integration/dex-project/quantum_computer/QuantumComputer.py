import numpy as np
from qiskit import QuantumCircuit, execute

class QuantumComputer:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)

    def add_hadamard_gate(self, qubit):
        self.circuit.h(qubit)

    def add_pauli_x_gate(self, qubit):
        self.circuit.x(qubit)

    def add_pauli_y_gate(self, qubit):
        self.circuit.y(qubit)

    def add_pauli_z_gate(self, qubit):
        self.circuit.z(qubit)

    def add_controlled_not_gate(self, control_qubit, target_qubit):
        self.circuit.cx(control_qubit, target_qubit)

    def add_measurement(self, qubit):
        self.circuit.measure(qubit, qubit)

    def execute_circuit(self, backend='qasm_simulator'):
        job = execute(self.circuit, backend)
        result = job.result()
        counts = result.get_counts(self.circuit)
        return counts

    def simulate(self, num_shots=1024):
        backend = 'qasm_simulator'
        counts = self.execute_circuit(backend)
        return counts

    def run_on_real_quantum_computer(self, backend='ibmq_qasm_simulator'):
        counts = self.execute_circuit(backend)
        return counts

# Example usage:
qc = QuantumComputer(2)
qc.add_hadamard_gate(0)
qc.add_controlled_not_gate(0, 1)
qc.add_measurement(0)
qc.add_measurement(1)

counts = qc.simulate()
print(counts)

# To run on a real quantum computer, replace 'qasm_simulator' with the name of the backend
# counts = qc.run_on_real_quantum_computer('ibmq_qasm_simulator')
