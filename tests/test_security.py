"""Offline end-to-end tests for the `security` package surface."""

import unittest

from security import SecurityManager


class TestSecurityManager(unittest.TestCase):
    """Offline tests for password-derived encryption and auth."""

    def setUp(self):
        """Build a manager with known credentials per test."""
        self.security_manager = SecurityManager("alice", "s3cret")

    def test_encryption_changes_plaintext(self):
        """Encrypted data differs from the plaintext."""
        encrypted = self.security_manager.encrypt("Sensitive Data")
        self.assertIsInstance(encrypted, bytes)
        self.assertNotEqual(encrypted, b"Sensitive Data")

    def test_decryption_roundtrip(self):
        """Decrypting a ciphertext restores the original string."""
        original = "Sensitive Data"
        encrypted = self.security_manager.encrypt(original)
        self.assertEqual(self.security_manager.decrypt(encrypted), original)

    def test_authenticate_user(self):
        """The manager accepts the right password and rejects wrong ones."""
        self.assertTrue(self.security_manager.authenticate_user("s3cret"))
        self.assertFalse(self.security_manager.authenticate_user("wrong"))


if __name__ == "__main__":
    unittest.main()
