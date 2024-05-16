import hashlib
import time
from collections import namedtuple
from typing import List, Optional

# Define a named tuple for transaction data
Transaction = namedtuple('Transaction', 'sender, receiver, amount, timestamp')

class Block:
    def __init__(self, index: int, transactions: List[Transaction], previous_hash: Optional[str] = None):
        self.index = index
        self.transactions = transactions
        self.timestamp = int(time.time())
        self.previous_hash = previous_hash or ""
        self.hash = self.calculate_hash()
        self.nonce = 0

    def calculate_hash(self) -> str:
        """Calculate the SHA-256 hash of the block."""
        block_string = str(self.index) + \
                       str(self.previous_hash) + \
                       str(self.timestamp) + \
                       str(self.nonce) + \
                       str(self.transactions)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int) -> None:
        """Mine the block by incrementing the nonce until the hash meets the difficulty requirement."""
        target = '0' * difficulty
        while self.hash[0:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 2
        self.pending_transactions = []

    def create_genesis_block(self) -> Block:
        """Create and return the genesis block."""
        return Block(0, [Transaction("genesis", "KOSASIH", 100, 0)], "0")

    def get_latest_block(self) -> Block:
        """Return the latest block in the chain."""
        return self.chain[-1]

    def add_transaction(self, transaction: Transaction) -> None:
        """Add a new transaction to the pending transactions list."""
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self) -> None:
        """Mine the pending transactions and add a new block to the chain."""
        latest_block = self.get_latest_block()
        new_block = Block(latest_block.index + 1, self.pending_transactions, latest_block.hash)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.pending_transactions = []

    def is_chain_valid(self) -> bool:
        """Check if the blockchain is valid by verifying the hashes and transactions."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True
