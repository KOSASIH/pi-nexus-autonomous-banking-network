import hashlib
from datetime import datetime
from typing import List

class Block:
    def __init__(self, index: int, timestamp: datetime, data: str, previous_hash: str, hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = hash

    @staticmethod
    def calculate_hash(index: int, timestamp: datetime, data: str, previous_hash: str):
        value = f"{index}{timestamp}{data}{previous_hash}".encode()
        return hashlib.sha256(value).hexdigest()

    def __str__(self):
        return f"Block {self.index} - {self.hash}"
