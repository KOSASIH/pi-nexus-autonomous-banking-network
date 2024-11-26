import hashlib
import json
from time import time

class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.difficulty = 4  # Difficulty for mining new blocks
        self.max_block_size = 1024  # Maximum block size in bytes

        # Create the genesis block
        self.create_block(previous_hash='1', transactions=[])

    def create_block(self, previous_hash, transactions):
        block = Block(
            index=len(self.chain) + 1,
            previous_hash=previous_hash,
            transactions=transactions
        )
        self.chain.append(block)
        self.current_transactions = []  # Reset current transactions
        return block

    def add_transaction(self, sender, recipient, amount):
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        }
        transaction_size = len(json.dumps(transaction).encode())
        
        if self.get_current_block_size() + transaction_size > self.max_block_size:
            raise Exception("Transaction exceeds maximum block size.")
        
        self.current_transactions.append(transaction)
        return self.get_last_block().index + 1  # Return the index of the block that will hold this transaction

    def get_last_block(self):
        return self.chain[-1] if self.chain else None

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check if the hash of the block is correct
            if current.hash != current.calculate_hash():
                return False

            # Check if the previous block's hash is correct
            if current.previous_hash != previous.hash:
                return False

        return True

    def get_current_block_size(self):
        return sum(len(json.dumps(tx).encode()) for tx in self.current_transactions)

    def mine_block(self, miner_address):
        if not self.current_transactions:
            raise Exception("No transactions to mine.")

        previous_block = self.get_last_block()
        previous_hash = previous_block.hash

        # Create a new block with the current transactions
        block = self.create_block(previous_hash, self.current_transactions)

        # Reward the miner
        self.add_transaction(sender="system", recipient=miner_address, amount=1)  # Reward for mining

        return block

    def get_chain(self):
        return [block.__dict__ for block in self.chain]

    def get_transaction_history(self):
        return self.current_transactions

    def get_block_by_index(self, index):
        if index < 0 or index >= len(self.chain):
            raise Exception("Block index out of range.")
        return self.chain[index]
