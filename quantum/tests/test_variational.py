# test_variational.py
import unittest
from quantum.variational_circuit import run_variational_algorithm

class TestVariationalAlgorithm(unittest.TestCase):
    def test_vqe_execution(self):
        num_qubits = 2
        initial_params = [0.1, 0.2]
        optimal_value, optimal_params = run_variational_algorithm(num_qubits, initial_params)
        self.assertIsInstance(optimal_value, float)  # Check if the optimal value is a float
        self.assertIsInstance(optimal_params, np.ndarray)  # Check if optimal params are in ndarray

if __name__ == '__main__':
    unittest.main()
