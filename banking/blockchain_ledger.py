import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, timestamp, data):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data_string = str(self.index) + self.previous_hash + str(self.timestamp) + str(self.data)
        return hashlib.sha256(data_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

    def create_genesis_block(self):
        return Block(0, "0", int(time.time()), "Genesis Block")

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self):
        if len(self.pending_transactions) < 1:
            return False
        new_block = Block(len(self.chain), self.chain[-1].hash, int(time.time()), self.pending_transactions)
        self.chain.append(new_block)
        self.pending_transactions = []
        return True

# Example usage:
blockchain = Blockchain()
blockchain.add_transaction("Transaction 1")
blockchain.add_transaction("Transaction 2")
blockchain.mine_block()
print(blockchain.chain)
