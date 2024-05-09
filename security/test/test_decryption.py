import unittest
from security.decryption import Decryption

class TestDecryption(unittest.TestCase):
def setUp(self):
        self.decryption = Decryption("mysecretkey")

    def test_decrypt(self):
        plain_text = "This is a secret message"
        cipher_text = self.decryption.encrypt(plain_text)
        decrypted_text = self.decryption.decrypt(cipher_text)
        self.assertEqual(plain_text, decrypted_text)
