import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class KeyManager:
    def __init__(self, key_size, ciphertext_size):
        self.key_size = key_size
        self.ciphertext_size = ciphertext_size
        self.he = None
        self.public_key = None
        self.private_key = None

    def generate_keys(self):
        # Generate a new RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        # Serialize the public key to a PEM file
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open('public_key.pem', 'wb') as f:
            f.write(pem)

        # Serialize the private key to a PEM file
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open('private_key.pem', 'wb') as f:
            f.write(pem)

        # Initialize the homomorphic encryption library
        self.he = HomomorphicEncryption(self.public_key, self.private_key)

    def load_keys(self):
        # Load the public key from a PEM file
        with open('public_key.pem', 'rb') as f:
            pem = f.read()
        self.public_key = serialization.load_pem_public_key(pem, backend=default_backend())

        # Load the private key from a PEM file
        with open('private_key.pem', 'rb') as f:
            pem = f.read()
        self.private_key = serialization.load_pem_private_key(pem, password=None, backend=default_backend())

        # Initialize the homomorphic encryption library
        self.he = HomomorphicEncryption(self.public_key, self.private_key)

    def get_public_key(self):
        return self.public_key

    def get_private_key(self):
        return self.private_key
