import hashlib

class Block:
    def __init__(self, transactions, previous_block_hash):
        self.transactions = transactions
        self.previous_block_hash = previous_block_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        # Calculate the block hash using SHA-256
        pass

class Blockchain:
    def __init__(self):
        self.chain = [Block([], '0' * 64)]  # Genesis block

    def add_block(self, block):
        self.chain.append(block)

    def validate_block(self, block):
        # Validate the block using digital signatures and hash functions
        pass

bc = Blockchain()
tx1 = {'from': 'Alice', 'to': 'Bob', 'amount': 10}
tx2 = {'from': 'Bob', 'to': 'Charlie', 'amount': 5}
block = Block([tx1, tx2], bc.chain[-1].hash)
bc.add_block(block)
print(bc.chain[-1].hash)
