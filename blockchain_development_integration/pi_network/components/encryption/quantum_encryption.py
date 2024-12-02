# quantum_encryption.py
from pqcrypto.kem import saber

def encrypt_data(data):
    public_key, secret_key = saber.keypair()
    ciphertext, shared_secret = saber.encrypt(public_key, data)
    return ciphertext, shared_secret

def decrypt_data(ciphertext, secret_key):
    return saber.decrypt(secret_key, ciphertext)
