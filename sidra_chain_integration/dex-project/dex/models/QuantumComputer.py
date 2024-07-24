import qiskit

class QuantumComputer:
    def __init__(self, backend):
        self.backend = backend

    def execute_circuit(self, circuit):
        job = qiskit.execute(circuit, self.backend)
        return job.result()

    def simulate_circuit(self, circuit):
        simulator = qiskit.Aer.get_backend('qasm_simulator')
        job = qiskit.execute(circuit, simulator)
        return job.result()
