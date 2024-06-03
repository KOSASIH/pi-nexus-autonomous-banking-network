import hashlib
import json
import time
from typing import Dict, List, Optional

class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, data: Dict, hash: str):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 2

    def create_genesis_block(self) -> Block:
        return Block(0, "0" * 64, int(time.time()), {"transaction_data": "Genesis Block"}, self.calculate_hash(0, "0" * 64, int(time.time()), {"transaction_data": "Genesis Block"}))

    def calculate_hash(self, index: int, previous_hash: str, timestamp: float, data: Dict) -> str:
        value = str(index) + previous_hash + str(timestamp) + json.dumps(data, sort_keys=True)
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    def add_block(self, data: Dict) -> None:
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, int(time.time()), data, None)
        new_block.hash = self.calculate_hash(new_block.index, new_block.previous_hash, new_block.timestamp, new_block.data)
        self.chain.append(new_block)

    def is_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != self.calculate_hash(current_block.index, current_block.previous_hash, current_block.timestamp, current_block.data):
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True
