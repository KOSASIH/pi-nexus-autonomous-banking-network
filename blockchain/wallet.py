import hashlib
import time

class Wallet:
    def __init__(self):
        self.private_key = hashlib.sha256(str(time.time()).encode('utf-8')).hexdigest()
        self.public_key = hashlib.sha256(self.private_key.encode('utf-8')).hexdigest()

    def generate_address(self):
        return hashlib.sha256(self.public_key.encode('utf-8')).hexdigest()

    def send_transaction(self, recipient, amount):
        transaction = Transaction(self.generate_address(), recipient, amount)
        return transaction
