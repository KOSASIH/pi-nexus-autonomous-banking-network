# sidra_chain_cryptography.py
import hashlib
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

class SidraChainCryptography:
    def __init__(self):
        # Generate a new RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

    def encrypt(self, data):
        # Encrypt data using the public key
        encrypted_data = self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted_data)

    def decrypt(self, encrypted_data):
        # Decrypt data using the private key
        encrypted_data = base64.b64decode(encrypted_data)
        decrypted_data = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data

    def sign(self, data):
        # Sign data using the private key
        signer = self.private_key.signer(
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        signature = signer.sign(data)
        return base64.b64encode(signature)

    def verify(self, data, signature):
        # Verify signature using the public key
        signature = base64.b64decode(signature)
        verifier = self.public_key.verifier(
            signature,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        verifier.verify(data)

    def generate_key_pair(self):
        # Generate a new RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def serialize_private_key(self):
        # Serialize the private key
        private_key_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return private_key_pem

    def serialize_public_key(self):
        # Serialize the public key
        public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )
        return public_key_pem

    def deserialize_private_key(self, private_key_pem):
        # Deserialize the private key
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )
        return private_key

    def deserialize_public_key(self, public_key_pem):
        # Deserialize the public key
        public_key = serialization.load_ssh_public_key(
            public_key_pem,
            backend=default_backend()
        )
        return public_key
