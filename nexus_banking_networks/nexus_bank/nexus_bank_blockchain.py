import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, transactions):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data_string = str(self.index) + self.previous_hash + str(self.transactions) + str(self.timestamp)
        return hashlib.sha256(data_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

    def create_genesis_block(self):
        return Block(0, "0", [])

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self):
        if len(self.pending_transactions) < 1:
            return False
        new_block = Block(len(self.chain), self.chain[-1].hash, self.pending_transactions)
        self.chain.append(new_block)
        self.pending_transactions = []
        return True

blockchain = Blockchain()

# Example usage:
blockchain.add_transaction({"amount": 10, "account_number": "1234567890"})
blockchain.add_transaction({"amount": 20, "account_number": "9876543210"})
blockchain.mine_block()
print(blockchain.chain)
