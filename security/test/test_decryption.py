"""Offline unit tests for :mod:`security.decryption`."""

import unittest

from cryptography.fernet import Fernet

from security.decryption import Decryption


class TestDecryption(unittest.TestCase):
    """Round-trip coverage for the Fernet-backed decryption wrapper."""

    def setUp(self):
        self.key = Fernet.generate_key()
        self.decryption = Decryption(self.key)
        self.cipher_suite = Fernet(self.key)

    def test_decrypt(self):
        """A Fernet ciphertext decrypts back to its original plain text."""
        plain_text = "This is a secret message"
        cipher_text = self.cipher_suite.encrypt(plain_text.encode("utf-8"))
        decrypted_text = self.decryption.decrypt(cipher_text)
        self.assertEqual(plain_text, decrypted_text)
