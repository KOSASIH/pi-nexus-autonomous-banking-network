import hashlib
import time
import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding

class PiNode:
    def __init__(self, node_id, private_key, public_key):
        self.node_id = node_id
        self.private_key = private_key
        self.public_key = public_key
        self.blockchain = []
        self.transaction_pool = []
        self.peers = []
        self.consensus_algorithm = 'proof-of-work'  # or 'proof-of-stake'
        self.difficulty_target = 4  # adjust difficulty for proof-of-work
        self.stake_weight = 1  # adjust stake weight for proof-of-stake

    def generate_key_pair(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def create_block(self, transactions, previous_block_hash):
        block = {
            'index': len(self.blockchain) + 1,
            'timestamp': int(time.time()),
            'transactions': transactions,
            'previous_block_hash': previous_block_hash,
            'node_id': self.node_id,
            'nonce': 0,
            'hash': self.calculate_block_hash(block)
        }
        return block

    def calculate_block_hash(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def add_block(self, block):
        self.blockchain.append(block)

    def get_blockchain(self):
        return self.blockchain

    def get_public_key(self):
        return self.public_key

    def sign_transaction(self, transaction):
        signer = self.private_key.signer(
            padding.PSS(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        signature = signer.sign(json.dumps(transaction, sort_keys=True).encode())
        return signature

    def verify_transaction(self, transaction, signature):
        verifier = self.public_key.verifier(
            signature,
            padding.PSS(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        try:
            verifier.verify(json.dumps(transaction, sort_keys=True).encode())
            return True
        except InvalidSignature:
            return False

    def encrypt_transaction(self, transaction):
        cipher = Cipher(algorithms.AES(self.private_key), modes.CBC(b'\00' * 16), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(json.dumps(transaction, sort_keys=True).encode()) + padder.finalize()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return encrypted_data

    def decrypt_transaction(self, encrypted_transaction):
        cipher = Cipher(algorithms.AES(self.private_key), modes.CBC(b'\00' * 16), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded_data = decryptor.update(encrypted_transaction) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        unpadded_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()
        return json.loads(unpadded_data.decode())

    def add_peer(self, peer):
        self.peers.append(peer)

    def get_peers(self):
        return self.peers

    def broadcast_transaction(self, transaction):
        for peer in self.peers:
            peer.add_transaction(transaction)

    def broadcast_block(self, block):
        for peer in self.peers:
            peer.add_block(block)

    def validate_transaction(self, transaction):
        # basic transaction validation (e.g., check sender and recipient, amount, etc.)
        return True

    def verify_block(self, block):
        # basic block verification (e.g., check block hash, transactions, etc.)
        return True

    def mine_block(self, transactions):
        if self.consensus_algorithm == 'proof-of-work':
            return self.mine_block_pow(transactions)
        elif self.consensus_algorithm == 'proof-of-stake':
            return self.mine_block_pos(transactions)

    def mine_block_pow(self, transactions):
        previous_block_hash = self.blockchain[-1]['block_hash'] if self.blockchain else '0' * 64
        nonce = 0
        while True:
            block = self.create_block(transactions, previous_block_hash)
            block['nonce'] = nonce
            block_hash = self.calculate_block_hash(block)
            if int(block_hash, 16) < 2 ** (256 - self.difficulty_target):
                self.add_block(block)
                return block
            nonce += 1

    def mine_block_pos(self, transactions):
        # proof-of-stake consensus algorithm
        # select a node to create a new block based on their stake weight
        nodes = self.peers + [self]
        nodes.sort(key=lambda node: node.stake_weight, reverse=True)
        for node in nodes:
            if node.stake_weight > 0:
                previous_block_hash = node.blockchain[-1]['block_hash'] if node.blockchain else '0' * 64
                block = node.create_block(transactions, previous_block_hash)
                node.add_block(block)
                return block
        return None

    def get_stake_weight(self):
        return self.stake_weight

    def set_stake_weight(self, stake_weight):
        self.stake_weight = stake_weight

    def get_transaction_pool(self):
        return self.transaction_pool

    def add_transaction(self, transaction):
        self.transaction_pool.append(transaction)

    def clear_transaction_pool(self):
        self.transaction_pool = []

    def get_node_id(self):
        return self.node_id

    def get_public_key(self):
        return self.public_key

    def get_private_key(self):
        return self.private_key
