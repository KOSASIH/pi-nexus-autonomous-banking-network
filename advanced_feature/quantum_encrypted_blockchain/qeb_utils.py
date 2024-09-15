import hashlib
import os

# Define function to generate quantum-resistant key pair
def generate_quantum_resistant_key_pair():
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Generate public key
    public_key = private_key.public_key()

    # Return public and private keys
    return public_key, private_key

# Define function to serialize public key
def serialize_public_key(public_key):
    # Serialize public key to PEM format
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # Return serialized public key
    return public_key_pem
