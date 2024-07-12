import numpy as np
from qiskit import QuantumCircuit, execute


class QuantumComputer:

    def __init__(self):
        self.circuit = QuantumCircuit(5, 5)

    def add_gate(self, gate, qubit):
        self.circuit.append(gate, [qubit])

    def run_simulation(self):
        job = execute(self.circuit, backend="qasm_simulator")
        result = job.result()
        return result.get_counts()


qc = QuantumComputer()
qc.add_gate(qc.h, 0)
qc.add_gate(qc.cx, [0, 1])
qc.add_gate(qc.measure, [1, 2])
print(qc.run_simulation())
