import unittest
from security import SecurityManager  # Assuming you have a SecurityManager class

class TestSecurityManager(unittest.TestCase):
    def setUp(self):
        self.security_manager = SecurityManager()

    def test_encryption(self):
        original_data = "Sensitive Data"
        encrypted_data = self.security_manager.encrypt(original_data)
        self.assertNotEqual(original_data, encrypted_data)

    def test_decryption(self):
        original_data = "Sensitive Data"
        encrypted_data = self.security_manager.encrypt(original_data)
        decrypted_data = self.security_manager.decrypt(encrypted_data)
        self.assertEqual(original_data, decrypted_data)

if __name__ == "__main__":
    unittest.main()
