import os
from cryptography.fernet import Fernet

class Decryption:
    def __init__(self, key):
        self.key = key
        self.cipher_suite = Fernet(self.key)

    def decrypt(self, cipher_text):
        plain_text = self.cipher_suite.decrypt(cipher_text).decode()
        return plain_text
