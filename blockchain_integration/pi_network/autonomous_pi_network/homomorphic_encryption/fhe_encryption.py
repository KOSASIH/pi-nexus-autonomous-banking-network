import numpy as np
from Pyfhel import Pyfhel

class FHEncryption:
    def __init__(self, context):
        self.context = context
        self.pyfhel = Pyfhel()

    def keygen(self):
        # Generate a public and private key pair for FHE
        self.pyfhel.keyGen()

    def encrypt(self, plaintext):
        # Encrypt the plaintext using FHE
        ciphertext = self.pyfhel.encrypt(plaintext)
        return ciphertext

    def decrypt(self, ciphertext):
        # Decrypt the ciphertext using FHE
        plaintext = self.pyfhel.decrypt(ciphertext)
        return plaintext

    def evaluate(self, ciphertext1, ciphertext2):
        # Evaluate a function on the encrypted data using FHE
        result = self.pyfhel.evalAdd(ciphertext1, ciphertext2)
        return result
