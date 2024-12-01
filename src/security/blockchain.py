import hashlib
import json
from time import time

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_data = []
        self.create_block(previous_hash='1', proof=100)

    def create_block(self, proof, previous_hash):
        block = Block(len(self.chain) + 1, previous_hash, time(), self.current_data, self.hash(block))
        self.current_data = []
        self.chain.append(block)
        return block

    def add_data(self, sender, recipient, amount):
        self.current_data.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block.index + 1

    @staticmethod
    def hash(block):
        block_string = json.dumps(block.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

# Example usage
if __name__ == "__main__":
    blockchain = Blockchain()
    blockchain.add_data("Alice", "Bob", 50)
    blockchain.add_data("Bob", "Charlie", 30)

    last_block = blockchain.create_block(proof=200, previous_hash=blockchain.last_block.hash)
    print(f"New Block: {last_block.__dict__}")
