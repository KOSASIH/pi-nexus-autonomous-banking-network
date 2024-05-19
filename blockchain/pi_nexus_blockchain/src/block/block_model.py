from datetime import datetime
from hashlib import sha256
from typing import Any, Union

from base58 import b58decode, b58encode


class BlockHeader:
    """
    A class representing the block header.

    Attributes:
        index (int): The block index.
        previous_hash (str): The hash of the previous block.
        timestamp (float): The timestamp when the block was created.
        data_hash (str): The hash of the block data.
        nonce (int): The nonce used for Proof-of-Work.
        hash (str): The hash of the block header.
    """

    def __init__(
        self,
        index: int,
        previous_hash: str,
        timestamp: float,
        data: Any,
        nonce: int = 0,
    ):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.data_hash = sha256(str(data).encode()).hexdigest()
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate the hash of the block header."""
        data = f"{self.index}{self.previous_hash}{self.timestamp}{self.data_hash}{self.nonce}".encode()
        return sha256(sha256(data).digest()).hexdigest()


class Block:
    """
    A class representing a block in the blockchain.

    Attributes:
        index (int): The block index.
        previous_hash (str): The hash of the previous block.
        timestamp (float): The timestamp when the block was created.
        data (Any): The data stored in the block.
        hash (str): The hash of the block.
        block_header (BlockHeader): The block header.
    """

    def __init__(self, index: int, previous_hash: str, data: Any, nonce: int = 0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = datetime.now().timestamp()
        self.data = data
        self.block_header = BlockHeader(
            index, previous_hash, self.timestamp, self.data, nonce
        )
        self.hash = self.block_header.hash

    def __str__(self):
        return f"Block {self.index} - Hash: {self.hash} - Previous Hash: {self.previous_hash}"
