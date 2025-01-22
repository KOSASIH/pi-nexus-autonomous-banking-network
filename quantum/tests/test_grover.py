# test_grover.py
import unittest
from quantum.grover_circuit import create_grover_circuit
from qiskit import Aer, execute

class TestGroverAlgorithm(unittest.TestCase):
    def setUp(self):
        self.num_qubits = 3
        self.marked_element = 5  # Example marked element

    def test_grover_circuit_creation(self):
        circuit = create_grover_circuit(self.num_qubits, self.marked_element)
        self.assertEqual(circuit.num_qubits, self.num_qubits)
        self.assertIn('cx', circuit.data[1][0].name)  # Check for CNOT gate

    def test_grover_execution(self):
        circuit = create_grover_circuit(self.num_qubits, self.marked_element)
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        self.assertIn('101', counts)  # Check if the marked element is in the results

if __name__ == '__main__':
    unittest.main()
