# homomorphic_encryption.py
import helib

class HomomorphicEncryption:
    def __init__(self, public_key):
        self.public_key = public_key

    def encrypt(self, plaintext):
        # encrypt plaintext using public key
        return helib.encrypt(plaintext, self.public_key)

    def decrypt(self, ciphertext):
        # decrypt ciphertext using private key
        return helib.decrypt(ciphertext, self.public_key)
