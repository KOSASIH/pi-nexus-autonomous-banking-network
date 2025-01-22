# test_hybrid_quantum.py
import unittest
from quantum.hybrid_quantum import hybrid_algorithm

class TestHybridQuantumAlgorithm(unittest.TestCase):
    def test_hybrid_algorithm(self):
        result = hybrid_algorithm()
        self.assertTrue(result['success'])  # Check if the hybrid algorithm was successful

if __name__ == '__main__':
    unittest.main()
