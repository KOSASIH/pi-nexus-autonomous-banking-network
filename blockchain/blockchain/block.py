from typing import Any
from datetime import datetime
from hashlib import sha256

class Block:
    """
    A class representing a block in the blockchain.

    Attributes:
        index (int): The block index.
        previous_hash (str): The hash of the previous block.
        timestamp (float): The timestamp when the block was created.
        data (Any): The data stored in the block.
        hash (str): The hash of the block.
    """

    def __init__(self, index: int, previous_hash: str, data: Any):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = datetime.now().timestamp()
        self.data = data
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate the hash of the block."""
        data = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}".encode()
        return sha256(sha256(data).digest()).hexdigest()
