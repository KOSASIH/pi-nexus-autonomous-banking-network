import hashlib

class SecureConsensusMechanism:
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

if __name__ == '__main__':
    scm = SecureConsensusMechanism()
    block1 = {'previous_hash': '0' * 64, 'transactions': [], 'nonce': 0, 'difficulty': 4}
    scm.add_block(block1)

    block2 = {'previous_hash': block1['hash'], 'transactions': [], 'nonce': 1, 'difficulty': 4}
    scm.add_block(block2)

    print(scm.verify_block(block2))
