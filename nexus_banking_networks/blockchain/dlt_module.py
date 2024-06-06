# dlt_module.py
import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

class DLTModule:
    def __init__(self):
        self.rsa_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )
        self.ledger = {}

    def add_block(self, block: dict) -> None:
        block_hash = hashlib.sha256(str(block).encode()).hexdigest()
        self.ledger[block_hash] = block

    def get_block(self, block_hash: str) -> dict:
        return self.ledger.get(block_hash)

    def verify_block(self, block: dict) -> bool:
        block_hash = hashlib.sha256(str(block).encode()).hexdigest()
        if block_hash in self.ledger:
            return True
        return False

    def sign_block(self, block: dict) -> bytes:
        signature = self.rsa_key.sign(str(block).encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
        return signature

    def verify_signature(self, block: dict, signature: bytes) -> bool:
        try:
            self.rsa_key.verify(signature, str(block).encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
            return True
        except InvalidSignature:
            return False
