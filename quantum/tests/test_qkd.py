# test_qkd.py
import unittest
from quantum.qkd import quantum_key_distribution

class TestQKD(unittest.TestCase):
    def test_qkd_protocol(self):
        key_length = 128
        key = quantum_key_distribution(key_length)
        self.assertEqual(len(key), key_length)  # Check if the key length is correct

if __name__ == '__main__':
    unittest.main()
