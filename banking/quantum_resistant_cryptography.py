import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class QuantumResistantCryptography:
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

    def encrypt(self, data):
        encrypted_data = self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashlib.SHA256()),
                algorithm=hashlib.SHA256(),
                label=None
            )
        )
        return encrypted_data

    def decrypt(self, encrypted_data):
        decrypted_data = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashlib.SHA256()),
                algorithm=hashlib.SHA256(),
                label=None
            )
        )
        return decrypted_data

# Example usage:
cryptography = QuantumResistantCryptography()
data = b"Hello, World!"
encrypted_data = cryptography.encrypt(data)
decrypted_data = cryptography.decrypt(encrypted_data)
print(decrypted_data)
