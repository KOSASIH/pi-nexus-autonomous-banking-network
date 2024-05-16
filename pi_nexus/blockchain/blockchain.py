# blockchain/blockchain.py
from typing import List

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.mining_reward = 10

    def create_genesis_block(self) -> Block:
        """Create the genesis block."""
        return Block(0, "0", datetime.datetime.now(), "Genesis Block", "0")

    def get_latest_block(self) -> Block:
        """Get the latest block in the blockchain."""
        return self.chain[-1]
