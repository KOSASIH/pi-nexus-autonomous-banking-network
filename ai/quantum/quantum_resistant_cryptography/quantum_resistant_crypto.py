import hashlib
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class QuantumResistantCrypto:
    def __init__(self):
        pass

    def generate_key_pair(self):
        private_key = ec.generate_private_key(
            ec.SECP256R1(),
            default_backend()
        )
        return private_key

    def serialize_private_key(self, private_key, password):
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(password)
        )
        return pem

    def deserialize_private_key(self, pem, password):
        private_key = serialization.load_pem_private_key(
            pem,
            password=password,
            backend=default_backend()
        )
        return private_key

    def encrypt_message(self, public_key, message):
        cipher = Cipher(algorithms.AES(os.urandom(16)), modes.GCM(os.urandom(12)), backend=default_backend())
        encryptor = cipher.encryptor()
        ct = encryptor.update(message) + encryptor.finalize()
        return ct

    def decrypt_message(self, private_key, ciphertext):
        cipher = Cipher(algorithms.AES(os.urandom(16)), modes.GCM(os.urandom(12)), backend=default_backend())
        decryptor = cipher.decryptor()
        pt = decryptor.update(ciphertext) + decryptor.finalize()
        return pt

    def sign_message(self, private_key, message):
        signer = private_key.signer(
            ec.ECDSA(hashes.SHA256())
        )
        signature = signer.finalize()
        return signature

    def verify_signature(self, public_key, message, signature):
        verifier = public_key.verifier(
            signature,
            ec.ECDSA(hashes.SHA256()),
            default_backend()
        )
        verifier.verify()

# Example usage
crypto = QuantumResistantCrypto()
private_key = crypto.generate_key_pair()
serialized_private_key = crypto.serialize_private_key(private_key, b"my_secret_password")
print(serialized_private_key)

public_key = private_key.public_key()
message = b"Hello, Quantum World!"
ciphertext = crypto.encrypt_message(public_key, message)
print(ciphertext.hex())

decrypted_message = crypto.decrypt_message(private_key, ciphertext)
print(decrypted_message)

signature = crypto.sign_message(private_key, message)
print(signature.hex())

crypto.verify_signature(public_key, message, signature)
