import unittest
from security.encryption import Encryption

class TestEncryption(unittest.TestCase):
    def setUp(self):
        self.encryption = Encryption("mysecretkey")

    def test_encrypt(self):
        plain_text = "This is a secret message"
        cipher_text = self.encryption.encrypt(plain_text)
        self.assertNotEqual(plain_text, cipher_text)

    def test_decrypt(self):
        plain_text = "This is a secret message"
        cipher_text = self.encryption.encrypt(plain_text)
        decrypted_text = self.encryption.decrypt(cipher_text)
        self.assertEqual(plain_text, decrypted_text)
