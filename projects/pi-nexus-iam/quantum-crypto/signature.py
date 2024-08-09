# signature.py

import os
import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

class Signature:
    def __init__(self, private_key: ec.EllipticCurvePrivateKey, hash_function: str = 'SHA-256'):
        self.private_key = private_key
        self.hash_function = hash_function

    def sign(self, message: bytes) -> bytes:
        """
        Signs a message using the private key and returns the signature.
        """
        hash_object = hashlib.new(self.hash_function)
        hash_object.update(message)
        digest = hash_object.digest()

        signature = self.private_key.sign(
            digest,
            ec.ECDSA(self.hash_function),
            default_backend()
        )

        return signature

    def verify(self, message: bytes, signature: bytes) -> bool:
        """
        Verifies a signature using the corresponding public key.
        """
        hash_object = hashlib.new(self.hash_function)
        hash_object.update(message)
        digest = hash_object.digest()

        public_key = self.private_key.public_key()
        try:
            public_key.verify(
                signature,
                digest,
                ec.ECDSA(self.hash_function),
                default_backend()
            )
            return True
        except ValueError:
            return False

    @staticmethod
    def generate_key_pair(curve: str = 'secp256r1') -> (ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey):
        """
        Generates a key pair using the specified curve.
        """
        curve = ec.SECP256R1() if curve == 'secp256r1' else ec.SECP384R1()
        private_key = ec.generate_private_key(curve, default_backend())
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def derive_key(private_key: ec.EllipticCurvePrivateKey, salt: bytes, info: bytes) -> ec.EllipticCurvePrivateKey:
        """
        Derives a new private key using the HKDF algorithm.
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=private_key.key_size // 8,
            salt=salt,
            info=info,
            backend=default_backend()
        )
        derived_key = hkdf.derive(private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        ))
        return ec.load_der_private_key(derived_key, backend=default_backend())

def main():
    # Generate a key pair
    private_key, public_key = Signature.generate_key_pair()

    # Create a signature object
    signature = Signature(private_key)

    # Sign a message
    message = b'Hello, World!'
    signature_bytes = signature.sign(message)
    print(f'Signature: {signature_bytes.hex()}')

    # Verify the signature
    is_valid = signature.verify(message, signature_bytes)
    print(f'Is valid: {is_valid}')

    # Derive a new private key
    salt = os.urandom(16)
    info = b'Pi-Nexus IAM System'
    derived_private_key = Signature.derive_key(private_key, salt, info)
    print(f'Derived private key: {derived_private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.TraditionalOpenSSL, encryption_algorithm=serialization.NoEncryption()).decode()}')

if __name__ == '__main__':
    main()
