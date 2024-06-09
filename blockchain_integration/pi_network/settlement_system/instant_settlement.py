import hashlib

class InstantSettlement:
    def __init__(self):
        self.blockchain = []

    def add_block(self, block):
        self.blockchain.append(block)

    def verify_block(self, block):
        previous_block = self.blockchain[-1]
        if block['previous_hash']!= previous_block['hash']:
            return False
        if not self.validate_proof_of_work(block):
            return False
        return True

    def validate_proof_of_work(self, block):
        difficulty = block['difficulty']
        nonce = block['nonce']
        hash = hashlib.sha256(str(nonce).encode() + str(difficulty).encode()).hexdigest()
        if hash[:difficulty]!= '0' * difficulty:
            return False
        return True

    def process_transaction(self, transaction):
        # Process transaction and add to block
        block = {'previous_hash': self.blockchain[-1]['hash'], 'transactions': [transaction], 'nonce': 0, 'difficulty': 4}
        self.add_block(block)
        return True

if __name__ == '__main__':
    isys = InstantSettlement()
    transaction = {'sender': 'Alice', 'receiver': 'Bob', 'amount': 10}
    isys.process_transaction(transaction)
