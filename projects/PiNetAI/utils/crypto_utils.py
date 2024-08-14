# crypto_utils.py

import hashlib
import hmac

def hash_string(string, algorithm):
    return getattr(hashlib, algorithm)(string.encode()).hexdigest()

def sign_string(string, secret_key, algorithm):
    return hmac.new(secret_key.encode(), string.encode(), algorithm).hexdigest()

def verify_signature(string, signature, secret_key, algorithm):
    expected_signature = sign_string(string, secret_key, algorithm)
    return signature == expected_signature

def encrypt_string(string, public_key):
    # Implement encryption using public key
    pass

def decrypt_string(string, private_key):
    # Implement decryption using private key
    pass

def generate_keypair():
    # Implement keypair generation
    pass
