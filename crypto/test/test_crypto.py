import unittest
from crypto_guard import CryptoGuard

class TestCryptoGuard(unittest.TestCase):
    def test_encrypt(self):
        # Test the encrypt method
        crypto_guard = CryptoGuard()
        encrypted_data = crypto_guard.encrypt("Hello, World!")
        self.assertNotEqual(encrypted_data, "Hello, World!")

    def test_decrypt(self):
        # Test the decrypt method
        crypto_guard = CryptoGuard()
        encrypted_data = crypto_guard.encrypt("Hello, World!")
        decrypted_data = crypto_guard.decrypt(encrypted_data)
        self.assertEqual(decrypted_data, "Hello, World!")

if __name__ == "__main__":
    unittest.main()
