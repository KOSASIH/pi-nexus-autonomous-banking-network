import hashlib
import random

class Wallet:
    def __init__(self):
        self.private_key = self.generate_private_key()
        self.public_key = self.generate_public_key(self.private_key)

    def generate_private_key(self):
        return hashlib.sha256(str(random.getrandbits(256)).encode('utf-8')).hexdigest()

    def generate_public_key(self, private_key):
        return hashlib.sha256(private_key.encode('utf-8')).hexdigest()

    def sign_transaction(self, sender, recipient, amount):
        signature = hashlib.sha256(f'{sender}{recipient}{amount}{self.private_key}'.encode('utf-8')).hexdigest()
        return signature

    def verify_signature(self, sender, recipient, amount, signature):
        return signature == hashlib.sha256(f'{sender}{recipient}{amount}{self.public_key}'.encode('utf-8')).hexdigest()
