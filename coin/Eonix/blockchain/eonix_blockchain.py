import hashlib
import time
import json
from eonix_cryptography import eonix_hash, eonix_verify
from eonix_network import EonixNetwork

class EonixBlockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.network = EonixNetwork()
        self.block_time = 10  # 10 seconds
        self.difficulty = 4  # adjustable difficulty for mining
        self.block_reward = 10  # reward for mining a block

    def create_genesis_block(self):
        # Create the genesis block
        genesis_block = {
            'index': 0,
            'previous_hash': '0' * 64,
            'timestamp': int(time.time()),
            'transactions': [],
            'nonce': 0,
            'hash': self.calculate_hash(0, '0' * 64, int(time.time()), [], 0)
        }
        self.chain.append(genesis_block)

    def calculate_hash(self, index, previous_hash, timestamp, transactions, nonce):
        # Calculate the hash of a block
        data = str(index) + previous_hash + str(timestamp) + str(transactions) + str(nonce)
        return eonix_hash(data)

    def add_transaction(self, transaction):
        # Add a transaction to the pending transactions list
        self.pending_transactions.append(transaction)

    def mine_block(self):
        # Mine a new block
        if len(self.pending_transactions) < 1:
            print("No transactions to mine.")
            return

        # Create a new block
        new_block = {
            'index': len(self.chain),
            'previous_hash': self.chain[-1]['hash'],
            'timestamp': int(time.time()),
            'transactions': self.pending_transactions,
            'nonce': 0
        }

        # Mine the block
        nonce = 0
        while True:
            new_block['nonce'] = nonce
            new_block['hash'] = self.calculate_hash(new_block['index'], new_block['previous_hash'], new_block['timestamp'], new_block['transactions'], new_block['nonce'])
            if self.validate_proof(new_block['hash'], self.difficulty):
                break
            nonce += 1

        # Add the block to the chain
        self.chain.append(new_block)
        self.pending_transactions = []

        # Reward the miner
        self.network.add_transaction({
            'from': 'eonix',
            'to': self.network.nodes[0],
            'amount': self.block_reward
        })

    def validate_proof(self, hash, difficulty):
        # Validate the proof of work
        return hash[:difficulty] == '0' * difficulty

    def validate_chain(self):
        # Validate the entire blockchain
        for i in range(1, len(self.chain)):
            block = self.chain[i]
            previous_block = self.chain[i - 1]
            if block['previous_hash'] != previous_block['hash']:
                return False
            if not self.validate_proof(block['hash'], self.difficulty):
                return False
        return True

    def get_chain(self):
        return self.chain

    def get_pending_transactions(self):
        return self.pending_transactions
