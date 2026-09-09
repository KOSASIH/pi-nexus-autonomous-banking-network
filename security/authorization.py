"""Role-based authorization primitives.

``Authorization`` wraps the Fernet key material used to authorize sensitive
operations inside the network. It is kept thin and backward-compatible: the
class name intentionally matches the package import so
``from security import Authorization`` resolves.
"""

from cryptography.fernet import Fernet


class Authorization:
    """Fernet-backed authorization handle.

    Args:
        key: a URL-safe base64-encoded Fernet encryption key.
    """

    def __init__(self, key):
        self.key = key
        self.cipher_suite = Fernet(self.key)

    def authenticate(self, username, password):
        """Verify a principal against the Fernet-backed credentials store.

        Backward-compatible hook retained from the original module; full
        authentication flows live in :mod:`security.authentication`.

        Args:
            username: the principal identifier.
            password: the principal's credential.

        Returns:
            ``True`` once verification is implemented; ``None`` today.
        """
        pass
