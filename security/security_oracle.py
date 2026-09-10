"""RSA-signed request oracle and its HTTP-facing API wrapper.

The oracle signs and verifies request payloads on behalf of the network
(`SecurityOracle`), while `SecurityOracleAPI` is the dictionary-shaped
surface consumed by HTTP handlers:

    oracle = SecurityOracle(private_key_path, public_key_path)
    api = SecurityOracleAPI(oracle)
    api.sign_data(data)            # -> {"signature": "<hex>"}
    api.verify_signature(data, s)  # -> {"message": "..."}

Both algorithms use RSASSA-PSS / SHA-256. The module imports nothing but
the standard library at load time: `cryptography` is resolved lazily so the
package stays importable (and testable) in constrained environments.
"""

import os


class SecurityOracle:
    """Signatures over data, backed by optional PEM key files."""

    def __init__(self, private_key_path=None, public_key_path=None):
        self.private_key_path = private_key_path
        self.public_key_path = public_key_path
        self.private_key = None
        self.public_key = None
        if private_key_path and os.path.exists(private_key_path):
            self._load_private_key()
        if public_key_path and os.path.exists(public_key_path):
            self._load_public_key()

    @staticmethod
    def _crypto():
        """Return the lazily imported ``cryptography`` serialization module."""
        from cryptography.hazmat.primitives import serialization

        return serialization

    def _load_private_key(self):
        """Load the RSA private key from ``private_key_path`` into memory."""
        serialization = self._crypto()
        with open(self.private_key_path, "rb") as key_file:
            self.private_key = serialization.load_pem_private_key(
                key_file.read(), password=None
            )

    def _load_public_key(self):
        """Load the RSA public key from ``public_key_path`` into memory."""
        serialization = self._crypto()
        with open(self.public_key_path, "rb") as key_file:
            self.public_key = serialization.load_pem_public_key(
                key_file.read()
            )

    @property
    def ready(self):
        """Return whether both signing and verification keys are loaded."""
        return self.private_key is not None and self.public_key is not None

    def generate_keys(self, key_size=2048):
        """Generate a fresh RSA keypair, persisting PEM files if paths are set.

        The keypair is always held in memory; the PEM files are written only
        when the oracle was constructed with paths.
        """
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(
            public_exponent=65537, key_size=key_size
        )
        serializers = serialization
        private_pem = key.private_bytes(
            encoding=serializers.Encoding.PEM,
            format=serializers.PrivateFormat.PKCS8,
            encryption_algorithm=serializers.NoEncryption(),
        )
        public_pem = key.public_key().public_bytes(
            encoding=serializers.Encoding.PEM,
            format=serializers.PublicFormat.SubjectPublicKeyInfo,
        )
        self.private_key = key
        self.public_key = key.public_key()
        if self.private_key_path:
            file_descriptor = os.open(
                self.private_key_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            with os.fdopen(file_descriptor, "wb") as key_file:
                key_file.write(private_pem)
        if self.public_key_path:
            with open(self.public_key_path, "wb") as key_file:
                key_file.write(public_pem)
        return {
            "private_key_path": self.private_key_path,
            "public_key_path": self.public_key_path,
            "key_size": key_size,
        }

    def sign_data(self, data):
        """RsA-PSS sign ``data`` and return the raw signature bytes."""
        if self.private_key is None:
            raise ValueError("SecurityOracle has no private key loaded")
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        return self.private_key.sign(
            data.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

    def verify_signature(self, data, signature):
        """Verify the RSA-PSS signature over ``data``."""
        if self.public_key is None:
            return False
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        try:
            self.public_key.verify(
                signature,
                data.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False


class SecurityOracleAPI:
    """Dictionary-shaped, JSON-friendly wrapper around `SecurityOracle`."""

    def __init__(self, oracle):
        self.oracle = oracle

    def generate_keys(self, key_size=2048):
        """Generate a keypair through the wrapped oracle."""
        result = self.oracle.generate_keys(key_size=key_size)
        return {"message": "Keys generated successfully", "details": result}

    def sign_data(self, data):
        """Return ``data`` signed, as a hex string."""
        if not self.oracle.ready:
            return {"error": "SecurityOracle has no keypair loaded"}
        signature = self.oracle.sign_data(data)
        return {"signature": signature.hex()}

    def verify_signature(self, data, signature):
        """Verify a hex signature against ``data``."""
        try:
            signature_bytes = bytes.fromhex(signature)
        except (TypeError, ValueError):
            return {"message": "Invalid signature"}
        if self.oracle.verify_signature(data, signature_bytes):
            return {"message": "Signature verified successfully"}
        return {"message": "Invalid signature"}
