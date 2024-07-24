import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.mining_reward = 10

    def create_genesis_block(self):
        return Block(0, '0', int(time.time()), 'Genesis Block', self.calculate_hash(0, '0', int(time.time()), 'Genesis Block'))

    def calculate_hash(self, index, previous_hash, timestamp, data):
        value = str(index) + str(previous_hash) + str(timestamp) + str(data)
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    def add_transaction(self, sender, recipient, amount):
        self.pending_transactions.append({'sender': sender, 'recipient': recipient, 'amount': amount})

    def mine_pending_transactions(self, miner):
        if len(self.pending_transactions) < 1:
            return False

        new_block = Block(len(self.chain), self.chain[-1].hash, int(time.time()), self.pending_transactions, self.calculate_hash(len(self.chain), self.chain[-1].hash, int(time.time()), self.pending_transactions))
        self.chain.append(new_block)

        self.pending_transactions = [{'sender': 'network', 'recipient': miner, 'amount': self.mining_reward}]
        return True
