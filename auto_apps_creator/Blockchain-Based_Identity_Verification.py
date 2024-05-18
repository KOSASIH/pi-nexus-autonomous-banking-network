import hashlib

from ecdsa import SigningKey, VerifyingKey


class IdentityVerifier:
    def __init__(self):
        self.signing_key = SigningKey.from_secret_exponent(123, curve=ecdsa.SECP256k1)
        self.verifying_key = self.signing_key.verifying_key

    def generate_identity(self, user_data):
        # Use elliptic curve cryptography to generate a unique identity
        # for each user
        identity = hashlib.sha256(user_data.encode()).hexdigest()
        return identity

    def verify_identity(self, identity, signature):
        # Use the verifying key to verify the signature
        return self.verifying_key.verify(signature, identity.encode())
