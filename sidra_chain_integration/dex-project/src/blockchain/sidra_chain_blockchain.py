# sidra_chain_blockchain.py
import hashlib
import time

class SidraChainBlockchain:
    def __init__(self):
        pass

    def create_block(self, index, previous_hash, transactions):
        # Create a block in the blockchain
        block = {
            'index': index,
            'previous_hash': previous_hash,
            'transactions': transactions,
            'timestamp': time.time(),
            'hash': self.calculate_hash(index, previous_hash, transactions)
        }
        return block

    def calculate_hash(self, index, previous_hash, transactions):
        # Calculate the hash of a block
        data = str(index) + previous_hash + str(transactions) + str(time.time())
        return hashlib.sha256(data.encode()).hexdigest()

    def add_block(self, blockchain, block):
        # Add a block to the blockchain
        blockchain.append(block)

    def validate_blockchain(self, blockchain):
        # Validate the blockchain
        for i in range(1, len(blockchain)):
            block = blockchain[i]
            previous_block = blockchain[i-1]
            if block['previous_hash']!= previous_block['hash']:
                return False
        return True
