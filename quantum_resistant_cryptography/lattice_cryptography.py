import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class LatticeCryptography:
    def __init__(self, dimension, modulus):
        self.dimension = dimension
        self.modulus = modulus
        self.private_key = self.generate_private_key()
        self.public_key = self.generate_public_key()

    def generate_private_key(self):
        # Generate a random lattice basis
        basis = np.random.randint(0, self.modulus, size=(self.dimension, self.dimension))
        return basis

    def generate_public_key(self):
        # Compute the public key from the private key
        public_key = np.dot(self.private_key, self.private_key.T) % self.modulus
        return public_key

    def encrypt(self, message):
        # Encrypt the message using the public key
        ciphertext = np.dot(message, self.public_key) % self.modulus
        return ciphertext

    def decrypt(self, ciphertext):
        # Decrypt the ciphertext using the private key
        message = np.dot(ciphertext, np.linalg.inv(self.private_key)) % self.modulus
        return message

    def serialize_public_key(self):
        # Serialize the public key to a byte string
        public_key_bytes = self.public_key.tobytes()
        return public_key_bytes

    def deserialize_public_key(self, public_key_bytes):
        # Deserialize the public key from a byte string
        public_key = np.frombuffer(public_key_bytes, dtype=np.int64)
        public_key = public_key.reshape((self.dimension, self.dimension))
        return public_key
