# qc.py
import qiskit
from qiskit import QuantumCircuit, execute

class EonixQC:
    def __init__(self):
        self.backend = qiskit.BasicAer.get_backend('qasm_simulator')

    def create_circuit(self, qubits, gates):
        circuit = QuantumCircuit(qubits)
        for gate in gates:
            circuit.append(gate, [qubits[gate[0]]])
        return circuit

    def execute_circuit(self, circuit):
        job = execute(circuit, self.backend)
        result = job.result()
        return result.get_counts(circuit)

    def simulate_quantum_computing(self, qubits, gates):
        circuit = self.create_circuit(qubits, gates)
        result = self.execute_circuit(circuit)
        return result
