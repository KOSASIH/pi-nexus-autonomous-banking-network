# blockchain.py
import hashlib
import json
from time import time

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.mining_reward = 1
        self.difficulty = 2

    def create_genesis_block(self):
        # ...

    def get_latest_block(self):
        # ...

    def add_transaction(self, sender, recipient, amount):
        # ...

    def mine_pending_transactions(self, miner):
        # ...

    def validate_chain(self):
        # ...
