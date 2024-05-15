# pi_nexus/blockchain.py
import hashlib

class Blockchain:
    def __init__(self) -> None:
        self.chain = []
        self.pending_transactions = []

    def add_block(self, transactions: list) -> None:
        block = {
            'index': len(self.chain) + 1,
            'timestamp': datetime.datetime.now(),
            'transactions': transactions,
            'previous_hash': self.chain[-1]['hash'] if self.chain else '0'
        }
        block['hash'] = self.calculate_hash(block)
        self.chain.append(block)

    def calculate_hash(self, block: dict) -> str:
        return hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()
