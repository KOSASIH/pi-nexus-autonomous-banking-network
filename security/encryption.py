import os
from cryptography.fernet import Fernet

class Encryption:
    def __init__(self, key):
        self.key = key
        self.cipher_suite = Fernet(self.key)

    def encrypt(self, data):
        cipher_text = self.cipher_suite.encrypt(data.encode())
        return cipher_text

    def generate_key(self):
        key = Fernet.generate_key()
        return key
