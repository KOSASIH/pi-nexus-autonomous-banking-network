# test_noise_simulation.py
import unittest
from quantum.quantum_noise import create_noise_model, run_circuit_with_noise
from qiskit import QuantumCircuit

class TestNoiseSimulation(unittest.TestCase):
    def test_noise_model_creation(self):
        noise_model = create_noise_model('depolarizing', 0.1)
        self.assertIsNotNone(noise_model)  # Check if the noise model is created

    def test_noise_application(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        noise_model = create_noise_model('depolarizing', 0.1)
        counts, statevector = run_circuit_with_noise(circuit, noise_model)
        self.assertIsInstance(counts, dict)  # Check if counts are returned as a dictionary

if __name__ == '__main__':
    unittest.main()
