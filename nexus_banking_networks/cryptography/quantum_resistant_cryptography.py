# quantum_resistant_cryptography.py
import hashlib

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


class QuantumResistantCryptography:

    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

    def encrypt(self, plaintext):
        ciphertext = self.public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashlib.sha256()),
                algorithm=hashlib.sha256(),
                label=None,
            ),
        )
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = self.private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashlib.sha256()),
                algorithm=hashlib.sha256(),
                label=None,
            ),
        )
        return plaintext


# Example usage:
qrc = QuantumResistantCryptography()
plaintext = b"Top secret message!"
ciphertext = qrc.encrypt(plaintext)
print("Ciphertext:", ciphertext)
decrypted_text = qrc.decrypt(ciphertext)
print("Decrypted text:", decrypted_text)
