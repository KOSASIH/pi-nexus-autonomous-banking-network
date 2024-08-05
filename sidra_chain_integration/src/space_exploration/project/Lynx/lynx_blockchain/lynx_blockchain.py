import hashlib
import time

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []

    def create_block(self, previous_hash, timestamp, transactions):
        block = {
            'index': len(self.chain) + 1,
            'previous_hash': previous_hash,
            'timestamp': timestamp,
            'transactions': transactions,
            'hash': self.hash_block(previous_hash, timestamp, transactions)
        }
        return block

    def hash_block(self, previous_hash, timestamp, transactions):
        block_string = str(previous_hash) + str(timestamp) + str(transactions)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def add_block(self, block):
        self.chain.append(block)

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self):
        if not self.pending_transactions:
            return False
        previous_hash = self.get_latest_block()['hash']
        timestamp = time.time()
        transactions = self.pending_transactions
        block = self.create_block(previous_hash, timestamp, transactions)
        self.add_block(block)
        self.pending_transactions = []
        return block

# Example usage:
blockchain = Blockchain()
blockchain.add_transaction({'from': 'A', 'to': 'B', 'amount': 10})
blockchain.add_transaction({'from': 'B', 'to': 'C', 'amount': 5})
block = blockchain.mine_block()
print(block)
