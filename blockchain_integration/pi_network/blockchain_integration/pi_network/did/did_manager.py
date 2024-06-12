import ujson
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class DIDManager:
    def __init__(self, private_key):
        self.private_key = private_key
        self.public_key = self.private_key.public_key()

    def generate_did(self):
        did = ujson.dumps({'publicKey': self.public_key.serialize().decode('utf-8')})
        return did

    def verify_did(self, did, signature):
        public_key = serialization.load_pem_public_key(did.encode('utf-8'), backend=default_backend())
        public_key.verify(signature, did.encode('utf-8'), padding.PKCS1v15(), hashes.SHA256())

# Example usage:
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
did_manager = DIDManager(private_key)
did = did_manager.generate_did()
print(did)
