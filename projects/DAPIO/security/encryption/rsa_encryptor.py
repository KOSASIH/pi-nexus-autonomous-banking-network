import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class RSAEncryptor:
    def __init__(self, public_key: str, private_key: str):
        self.public_key = serialization.load_pem_public_key(public_key.encode(), backend=default_backend())
        self.private_key = serialization.load_pem_private_key(private_key.encode(), password=None, backend=default_backend())

    def encrypt(self, plaintext: bytes) -> bytes:
        encrypted_data = self.public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted_data)

    def decrypt(self, ciphertext: bytes) -> bytes:
        decrypted_data = self.private_key.decrypt(
            base64.b64decode(ciphertext),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data
