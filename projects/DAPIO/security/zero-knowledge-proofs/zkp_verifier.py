import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class ZKPVerifier:
    def __init__(self, public_key: str):
        self.public_key = serialization.load_pem_public_key(public_key.encode(), backend=default_backend())

    def verify(self, statement: str, proof: bytes) -> bool:
        hash_value = hashlib.sha256(statement.encode()).digest()
        signature = ec.ECDSA(self.public_key).verify(proof, hash_value)
        return signature.valid
