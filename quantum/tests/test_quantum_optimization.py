import unittest
import numpy as np
from quantum.algorithms.quantum_optimization import QuantumOptimizer

class TestQuantumOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test parameters for the Quantum Optimizer."""
        self.n_qubits = 4  # Number of qubits for the optimization problem
        self.p = 2         # Number of layers in the QAOA circuit
        self.optimizer = QuantumOptimizer(self.n_qubits, self.p)

    def test_create_qaoa_circuit(self):
        """Test the creation of the QAOA circuit."""
        gamma = [0.1, 0.2]  # Example gamma parameters
        beta = [0.3, 0.4]   # Example beta parameters
        circuit = self.optimizer.create_qaoa_circuit(gamma, beta)
        self.assertEqual(circuit.num_qubits, self.n_qubits, "Circuit should have the correct number of qubits.")

    def test_objective_function(self):
        """Test the objective function with known parameters."""
        params = [0.1, 0.2, 0.3, 0.4]  # Example parameters (gamma and beta)
        objective_value = self.optimizer.objective_function(params)
        self.assertIsInstance(objective_value, float, "Objective function should return a float value.")

    def test_optimize(self):
        """Test the optimization process."""
        best_params, best_score = self.optimizer.optimize()
        self.assertIsInstance(best_params, np.ndarray, "Best parameters should be a numpy array.")
        self.assertIsInstance(best_score, float, "Best score should be a float value.")
        self.assertGreaterEqual(best_score, 0, "Best score should be non-negative.")

    def test_invalid_parameters(self):
        """Test optimization with invalid parameters."""
        with self.assertRaises(ValueError):
            self.optimizer.optimize()  # Should raise an error if no valid parameters are set

if __name__ == "__main__":
    unittest.main()
