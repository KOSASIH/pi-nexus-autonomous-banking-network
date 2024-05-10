import hashlib
import json
from time import time
from typing import Any


class Transaction:
    def __init__(self, sender: str, receiver: str, amount: int):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount

    def to_dict(self) -> dict:
        return {"sender": self.sender, "receiver": self.receiver, "amount": self.amount}


class Block:
    def __init__(
        self,
        index: int,
        previous_hash: str,
        timestamp: int,
        transactions: list[Transaction],
        nonce: int = 0,
    ):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_data = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_data).hexdigest()

    def mine_block(self, difficulty: int):
        prefix = "0" * difficulty
        while not self.hash.startswith(prefix):
            self.nonce += 1
            self.hash = self.calculate_hash()
