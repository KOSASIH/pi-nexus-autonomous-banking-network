import numpy as np
from qiskit import QuantumCircuit, execute, Aer

class QuantumSimulation:
    def __init__(self, num_qubits, num_clbits):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.circuit = QuantumCircuit(num_qubits, num_clbits)

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

    def add_measurement(self, qubit, clbit):
        self.circuit.measure(qubit, clbit)

    def simulate(self, backend='qasm_simulator'):
        simulator = Aer.get_backend(backend)
        job = execute(self.circuit, simulator)
        result = job.result()
        counts = result.get_counts(self.circuit)
        return counts

    def plot_simulation(self, counts):
        import matplotlib.pyplot as plt
        plt.bar(counts.keys(), counts.values())
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.title('Quantum Simulation Results')
        plt.show()

# Example usage
simulation = QuantumSimulation(2, 2)
simulation.add_hadamard_gate(0)
simulation.add_controlled_not_gate(0, 1)
simulation.add_measurement(0, 0)
simulation.add_measurement(1, 1)
counts = simulation.simulate()
simulation.plot_simulation(counts)
