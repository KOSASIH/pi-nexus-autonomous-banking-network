# quantum-crypto-module/gen_keys.py
import os
import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

def generate_keys():
    # Generate RSA key pair
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend()
    )

    # Serialize private key
    private_key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Serialize public key
    public_key_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH
    )

    # Save keys to files
    with open('private_key.pem', 'wb') as f:
        f.write(private_key_pem)

    with open('public_key.pem', 'wb') as f:
        f.write(public_key_pem)

    return private_key_pem, public_key_pem

generate_keys()
