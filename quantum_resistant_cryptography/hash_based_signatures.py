import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

class HashBasedSignatures:
    def __init__(self, hash_function):
        self.hash_function = hash_function

    def sign(self, message, private_key):
        # Compute the hash of the message
        message_hash = self.hash_function(message)

        # Sign the hash using the private key
        signature = private_key.sign(message_hash, padding.PSS(mgf=padding.MGF1(self.hash_function), salt_length=padding.PSS.MAX_LENGTH), self.hash_function)

        return signature

    def verify(self, message, signature, public_key):
        # Compute the hash of the message
        message_hash = self.hash_function(message)

        # Verify the signature using the public key
        public_key.verify(signature, message_hash, padding.PSS(mgf=padding.MGF1(self.hash_function), salt_length=padding.PSS.MAX_LENGTH), self.hash_function)

    def serialize_public_key(self, public_key):
        # Serialize the public key to a byte string
        public_key_bytes = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        return public_key_bytes

    def deserialize_public_key(self, public_key_bytes):
        # Deserialize the public key from a byte string
        public_key = serialization.load_pem_public_key(public_key_bytes, backend=default_backend())
        return public_key
