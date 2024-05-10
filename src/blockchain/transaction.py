import hashlib
import json
from typing import Any


class Transaction:
    def __init__(self, sender: str, receiver: str, amount: int):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = time()
        self.signature = None

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }

    def calculate_hash(self) -> str:
        transaction_data = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(transaction_data).hexdigest()

    def sign_transaction(self, private_key: str):
        self.signature = self.calculate_hash()
        # Sign the transaction using the private key
        # This is a placeholder for the actual signing process
        self.signature = private_key
