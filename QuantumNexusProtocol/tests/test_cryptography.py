import unittest
from src.cryptography.secure_hashing import SecureHashing

class TestCryptography(unittest.TestCase):
    def setUp(self):
        self.crypto = SecureHashing()

    def test_hash_generation(self):
        hash_value = self.crypto.generate_hash("test_data")
        self.assertIsNotNone(hash_value)

    def test_verify_hash(self):
        hash_value = self.crypto.generate_hash("test_data")
        result = self.crypto.verify_hash("test_data ", hash_value)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
