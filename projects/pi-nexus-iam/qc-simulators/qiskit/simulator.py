from qiskit import QuantumCircuit, Aer

class QiskitSimulator:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)
        self.backend = Aer.get_backend('qasm_simulator')

    def add_gate(self, gate: str, qubit: int):
        if gate == 'h':
            self.circuit.h(qubit)
        elif gate == 'cx':
            self.circuit.cx(qubit, qubit + 1)
        else:
            raise ValueError(f"Invalid gate: {gate}")

    def run(self):
        job = self.backend.run(self.circuit)
        result = job.result()
        return result.get_counts(self.circuit)
