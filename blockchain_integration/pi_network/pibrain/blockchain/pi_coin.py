import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class PiCoin:
    def __init__(self, public_key, private_key):
        self.public_key = public_key
        self.private_key = private_key

    def generate_address(self):
        public_key_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        address = hashlib.sha256(public_key_bytes).hexdigest()[:20]
        return address

    def sign_transaction(self, transaction):
        # TO DO: implement transaction signing logic
        pass

    def verify_transaction(self, transaction, signature):
        # TO DO: implement transaction verification logic
        pass

class Transaction:
    def __init__(self, sender, recipient, amount):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount

    def to_dict(self):
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount
        }

# Example usage:
public_key = serialization.load_pem_public_key(b'public_key_pem', backend=default_backend())
private_key = serialization.load_pem_private_key(b'private_key_pem', password=None, backend=default_backend())

pi_coin = PiCoin(public_key, private_key)
address = pi_coin.generate_address()
print(f'Generated address: {address}')

transaction = Transaction(address, 'recipient_address', 10)
print(f'Transaction: {transaction.to_dict()}')
