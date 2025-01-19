# test_shor.py
import unittest
from quantum.shor_circuit import shor_algorithm

class TestShorAlgorithm(unittest.TestCase):
    def test_shor_algorithm(self):
        number = 15  # Example number to factor
        factors = shor_algorithm(number)
        self.assertIn(3, factors)
        self.assertIn(5, factors)

if __name__ == '__main__':
    unittest.main()
