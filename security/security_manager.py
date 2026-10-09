"""Authenticated user session and data-at-rest encryption.

`SecurityManager` is the single security context a wallet builds from a
user id and password:

    manager = SecurityManager(user_id, password)
    ciphertext = manager.encrypt("Sensitive Data")
    manager.decrypt(ciphertext)          # -> "Sensitive Data"
    manager.authenticate_user(password)  # -> True / False

The key is derived with scrypt (per-instance random salt), so two managers
never share a key even for identical credentials, and ciphertexts are
authenticated with AES-GCM (a nonce is stored with every message, so each
encryption uses fresh randomness).
"""

import hashlib
import hmac
import os


class SecurityManager:
    """Password-derived encryption and authentication context."""

    def __init__(self, user_id=None, password=None):
        self.user_id = user_id
        self.password = password
        self.salt = hashlib.sha256(os.urandom(32)).hexdigest()

    def derive_key(self, password=None):
        """Derive a 32-byte scrypt key for the given password.

        When `password` is omitted the key is derived from the password the
        manager was constructed with, which lets `authenticate_user` compare
        two derivations deterministically.
        """
        secret = password if password is not None else self.password
        if not secret:
            raise ValueError("no password available for key derivation")
        return hashlib.scrypt(
            secret.encode("utf-8"),
            salt=self.salt.encode("utf-8"),
            n=2 ** 14,
            r=8,
            p=1,
            dklen=32,
        )

    @staticmethod
    def _crypto():
        """Return the lazily imported ``cryptography`` primitives module."""
        from cryptography.hazmat.primitives import ciphers

        return ciphers

    def encrypt_data(self, data):
        """Encrypt a string to bytes with AES-GCM (fresh nonce per call)."""
        iv = os.urandom(12)
        ciphers = self._crypto()
        encryptor = ciphers.Cipher(
            ciphers.algorithms.AES(self.derive_key()), ciphers.modes.GCM(iv)
        ).encryptor()
        data_bytes = data.encode("utf-8")
        ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
        return iv + ciphertext + encryptor.tag

    def decrypt_data(self, ciphertext):
        """Decrypt a value produced by `encrypt_data` (authenticated)."""
        iv, body = ciphertext[:12], ciphertext[12:]
        ciphertext_body, tag = body[:-16], body[-16:]
        ciphers = self._crypto()
        decryptor = ciphers.Cipher(
            ciphers.algorithms.AES(self.derive_key()),
            ciphers.modes.GCM(iv, tag),
        ).decryptor()
        plaintext = decryptor.update(ciphertext_body) + decryptor.finalize()
        return plaintext.decode("utf-8")

    def encrypt(self, plaintext):
        """Encrypt a string and return the authenticated ciphertext."""
        return self.encrypt_data(plaintext)

    def decrypt(self, ciphertext):
        """Decrypt the authenticated ciphertext back to a string."""
        return self.decrypt_data(ciphertext)

    def authenticate_user(self, password):
        """Return whether ``password`` matches the manager's credentials."""
        if not password or not self.password:
            return False
        try:
            current = self.derive_key()
            candidate = self.derive_key(password)
            return hmac.compare_digest(current, candidate)
        except ValueError:
            return False
