import hashlib
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class PiNetwork:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.nodes = set()

    def add_node(self, node_id):
        self.nodes.add(node_id)

    def remove_node(self, node_id):
        self.nodes.remove(node_id)

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self, node_id):
        if not self.pending_transactions:
            return None

        block = {
            'index': len(self.chain) + 1,
            'timestamp': int(time.time()),
            'transactions': self.pending_transactions,
            'previous_hash': self.chain[-1]['hash'] if self.chain else '0',
            'node_id': node_id
        }

        block_hash = self.calculate_block_hash(block)
        block['hash'] = block_hash

        self.chain.append(block)
        self.pending_transactions = []

        return block

    def calculate_block_hash(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block['previous_hash'] != previous_block['hash']:
                return False

            if not self.validate_block_transactions(current_block):
                return False

        return True

    def validate_block_transactions(self, block):
        for transaction in block['transactions']:
            if not self.validate_transaction(transaction):
                return False

        return True

    def validate_transaction(self, transaction):
        # TO DO: implement transaction validation logic
        pass

pi_network = PiNetwork()
