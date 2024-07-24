import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data_string = str(self.index) + self.previous_hash + str(self.transactions) + str(self.timestamp)
        return hashlib.sha256(data_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.mining_reward = 10

    def create_genesis_block(self):
        return Block(0, "0", [])

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self, miner_reward_address):
        if not self.pending_transactions:
            return False
        new_block = Block(len(self.chain), self.get_latest_block().hash, self.pending_transactions)
        self.chain.append(new_block)
        self.pending_transactions = [
            {
                "from": "network",
                "to": miner_reward_address,
                "amount": self.mining_reward
            }
        ]
        return True
