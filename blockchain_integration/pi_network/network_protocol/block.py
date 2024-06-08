class Block:
    def __init__(self, index, previous_hash, transactions, timestamp, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = str(self.index) + self.previous_hash + str(self.timestamp) + str(self.nonce) + str(self.transactions)
        return hashlib.sha256(block_string.encode()).hexdigest()
