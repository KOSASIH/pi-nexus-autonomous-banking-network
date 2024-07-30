# wallet.py
import hashlib
import random

class Wallet:
    def __init__(self):
        self.private_key = self.generate_private_key()
        self.public_key = self.generate_public_key()

    def generate_private_key(self):
        return hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()

    def generate_public_key(self):
        return hashlib.sha256(self.private_key.encode()).hexdigest()

    def sign_transaction(self, transaction):
        return hashlib.sha256((transaction + self.private_key).encode()).hexdigest()

    def verify_transaction(self, transaction, signature):
        return hashlib.sha256((transaction + self.public_key).encode()).hexdigest() == signature
