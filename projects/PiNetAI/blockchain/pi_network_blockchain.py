import os
import json
import hashlib
import time
from typing import List, Dict
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: int, data: Dict[str, str], public_key: str):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.public_key = public_key
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        data_string = json.dumps(self.data, sort_keys=True)
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{data_string}{self.public_key}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_data = []
        self.public_key = self.generate_public_key()

    def create_genesis_block(self) -> Block:
        return Block(0, "0", int(time.time()), {"genesis": "block"}, self.public_key)

    def generate_public_key(self) -> str:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        ).decode()

    def add_block(self, data: Dict[str, str]) -> None:
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, int(time.time()), data, self.public_key)
        self.chain.append(new_block)

    def validate_chain(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

    def get_latest_block(self) -> Block:
        return self.chain[-1]

# Example usage:
blockchain = Blockchain()
blockchain.add_block({"transaction": "example"})
print(blockchain.get_latest_block().hash)
