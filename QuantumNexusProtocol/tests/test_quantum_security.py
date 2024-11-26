import unittest
from src.quantum.quantum_security import QuantumSecurity

class TestQuantumSecurity(unittest.TestCase):
    def setUp(self):
        self.qs = QuantumSecurity()

    def test_key_distribution(self):
        key = self.qs.distribute_key()
        self.assertIsNotNone(key)

    def test_randomness(self):
        randomness = self.qs.generate_randomness()
        self.assertEqual(len(randomness), 32)  # Assuming 32 bytes of randomness

if __name__ == '__main__':
    unittest.main()
