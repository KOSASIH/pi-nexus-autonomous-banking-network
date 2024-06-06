import hashlib
from ecdsa import SigningKey, VerifyingKey

class BBL:
    def __init__(self, private_key_path, public_key_path):
        self.private_key = SigningKey.from_pem(open(private_key_path, 'rb').read())
        self.public_key = VerifyingKey.from_pem(open(public_key_path, 'rb').read())

    def create_block(self, transactions_data):
        block_data = {'transactions': transactions_data, 'timestamp': int(time.time())}
        block_hash = hashlib.sha256(dumps(block_data).encode()).hexdigest()
        signature = self.private_key.sign(block_hash.encode())
        return block_data, signature

    def verify_block(self, block_data, signature):
        block_hash = hashlib.sha256(dumps(block_data).encode()).hexdigest()
        self.public_key.verify(signature, block_hash.encode())
