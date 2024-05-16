# block.py

import hashlib
import time
from typing import List, Dict

class Block:
    """
    A single block in the blockchain, representing a unit of data and its associated metadata.
    """

    def __init__(self, index: int, previous_hash: str, transactions: List[Dict], timestamp: float = None):
        """
        Initialize a new block.

        :param index: The block's index in the blockchain.
        :param previous_hash: The hash of the previous block.
        :param transactions: A list of transactions to be included in the block.
        :param timestamp: The timestamp of the block (defaults to current time).
        """
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """
        Calculate the block's hash using a combination of its attributes and a cryptographically secure hash function.
        """
        data = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}".encode()
        return hashlib.sha3_256(data).hexdigest()

    def __repr__(self) -> str:
        return f"Block {self.index} - {self.hash}"

    def to_dict(self) -> Dict:
        """
        Convert the block to a dictionary representation.
        """
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "hash": self.hash
        }
