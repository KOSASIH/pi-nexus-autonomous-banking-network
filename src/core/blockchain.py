import hashlib
import json
from datetime import datetime
from cryptography import generate_public_key, generate_private_key, sign_transaction, verify_signature
from services.analytics import analyze_transactions

class Transaction:
    def __init__(self, sender_id, receiver_id, amount):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.amount = amount
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.signature = None

    def sign_transaction(self, private_key):
        self.signature = sign_transaction(self.to_json(), private_key)

    def to_json(self):
        return json.dumps(self.__dict__)

class Block:
    def __init__(self, index, previous_hash, timestamp, transactions, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.hash = hash

    @staticmethod
    def calculate_hash(index, previous_hash, timestamp, transactions, nonce=0):
        data = f"{index}{previous_hash}{timestamp}{json.dumps(transactions)}{nonce}".encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    def mine_block(self, difficulty):
        leading_zeros = '0' * difficulty
        while self.hash[:difficulty] != leading_zeros:
            self.nonce += 1
            self.hash = self.calculate_hash(self.index, self.previous_hash, self.timestamp, self.transactions, self.nonce)

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 4
        self.pending_transactions = []

    def create_genesis_block(self):
        return Block(0, '0' * 64, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), [], '0' * 64)

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self):
        block = Block(len(self.chain), self.get_latest_block().hash, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.pending_transactions, None)
        block.mine_block(self.difficulty)
        self.chain.append(block)
        self.pending_transactions = []

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash(current_block.index, previous_block.hash, current_block.timestamp, current_block.transactions, current_block.nonce):
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def to_json(self):
        return json.dumps([block.__dict__ for block in self.chain], indent=4)

class Consensus:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def reach_consensus(self):
        while True:
            if not self.blockchain.is_valid():
                raise Exception("Blockchain is invalid")

            if len(self.blockchain.chain) < 2:
                continue

            if self.blockchain.chain[-1].index % 10 == 0:
                self.blockchain.mine_pending_transactions()
                analyze_transactions(self.blockchain)

            new_chain = self.request_new_chain()
            if len(new_chain) > len(self.blockchain.chain):
                self.blockchain.chain = new_chain

    def request_new_chain(self):
        # TODO: implement logic to request new chain from peers
        pass

    def is_new_chain_valid(self, new_chain):
        if len(new_chain) <= len(self.blockchain.chain):
            return False

        current_index = len(self.blockchain.chain) - 1
        while current_index >= 0:
            block = new_chain[current_index]
            if block.hash != block.calculate_hash(block.index, block.previous_hash, block.timestamp, block.transactions, block.nonce):
                return False

            current_index -= 1

        return True
