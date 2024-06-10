import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class BlockchainIdentityVerification:
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

    def generate_identity(self, user_data):
        identity_hash = hashlib.sha256(user_data.encode()).hexdigest()
        encrypted_identity = self.public_key.encrypt(
            identity_hash.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashlib.SHA256()),
                algorithm=hashlib.SHA256(),
                label=None
            )
        )
        return encrypted_identity

    def verify_identity(self, encrypted_identity):
        decrypted_identity = self.private_key.decrypt(
            encrypted_identity,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashlib.SHA256()),
                algorithm=hashlib.SHA256(),
                label=None
            )
        )
        return decrypted_identity.decode()

# Example usage:
blockchain_identity_verification = BlockchainIdentityVerification()
user_data = "John Doe"
encrypted_identity = blockchain_identity_verification.generate_identity(user_data)
verified_identity = blockchain_identity_verification.verify_identity(encrypted_identity)
print(verified_identity)
