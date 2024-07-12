# quantum_computing.py
import cirq
import numpy as np


class QuantumComputer:

    def __init__(self):
        self.qubits = [cirq.LineQubit(i) for i in range(5)]
        self.circuit = cirq.Circuit()

    def add_gates(self):
        self.circuit.append(cirq.H(self.qubits[0]))
        self.circuit.append(cirq.CNOT(self.qubits[0], self.qubits[1]))
        self.circuit.append(cirq.measure(self.qubits, key="result"))

    def run_circuit(self):
        simulator = cirq.Simulator()
        result = simulator.run(self.circuit, repetitions=1000)
        return result.histogram(key="result")

    def analyze_results(self, results):
        counts = results.get("result", {})
        probabilities = {k: v / sum(counts.values()) for k, v in counts.items()}
        return probabilities
