import hashlib
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class EndToEndEncryption:
    def __init__(self, private_key_path, public_key_path):
        self.private_key = serialization.load_pem_private_key(
            open(private_key_path, 'rb').read(),
            password=None,
            backend=default_backend()
        )
        self.public_key = serialization.load_pem_public_key(
            open(public_key_path, 'rb').read(),
            backend=default_backend()
        )

    def encrypt(self, data):
        encrypted_data = self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data

    def decrypt(self, encrypted_data):
        decrypted_data = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data

# Example usage:
private_key_path = 'path/to/private_key.pem'
public_key_path = 'path/to/public_key.pem'
e2e_encryption = EndToEndEncryption(private_key_path, public_key_path)

data = b'Hello, Universe!'
encrypted_data = e2e_encryption.encrypt(data)
print(encrypted_data)

decrypted_data = e2e_encryption.decrypt(encrypted_data)
print(decrypted_data)
