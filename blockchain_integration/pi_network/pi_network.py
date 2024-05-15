# pi_network.py

import hashlib
import time
import json
from flask import Flask, request, jsonify
from flask_api import APIException

app = Flask(__name__)

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 2

    def create_genesis_block(self):
        return self.create_new_block([], '0')

    def create_new_block(self, transactions, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': transactions,
            'previous_hash': previous_hash,
            'miner': '',
            'proof': ''
        }

        new_block = self.mine_block(block)
        return new_block

    def mine_block(self, block):
        new_proof = self.find_proof(block)
        if new_proof is None:
            return None

        block['proof'] = new_proof
        return block

    def find_proof(self, block):
        new_proof = block['index']
        check_hash = False

        while not check_hash:
            new_proof += 1
            new_block = block.copy()
            new_block['proof'] = new_proof
            new_hash = self.calculate_hash(new_block)

            if new_hash[:self.difficulty] == '0' * self.difficulty:
                check_hash = True
                return new_proof

        return None

    def calculate_hash(self, block):
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block['previous_hash'] != self.calculate_hash(previous_block):
                return False

            if not self.validate_proof(current_block):
                return False

        return True

    def validate_proof(self, block):
        previous_hash = self.calculate_hash(self.chain[block['index'] - 1])
        guess = f"{previous_hash}{block['transactions']}{block['miner']}".encode()
        hash = hashlib.sha256(guess).hexdigest()

        return hash[:self.difficulty] == '0' * self.difficulty

    def add_transaction(self, sender, recipient, amount):
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        }

        if not self.validate_transaction(transaction):
            raise APIException(f"Invalid transaction: {transaction}")

        self.mine_pending_transactions(f"New transaction from {sender} to {recipient} for {amount} coins")

    def validate_transaction(self, transaction):
        # Add validation logic here
        return True

    def mine_pending_transactions(self, miner):
        if not self.pending_transactions:
            raise APIException("No transactions to mine")

        last_block = self.chain[-1]
        new_block = self.create_new_block(self.pending_transactions, last_block['hash'])
        new_block['miner'] = miner

        self.chain.append(new_block)
        self.pending_transactions = []

        return new_block

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)

    def replace_chain(self):
        new_chain = None

        max_length = len(self.chain)
        for node in self.nodes:
            response = requests.get(f"{node}/get_chain")

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                if length > max_length and self.validate_chain(chain):
                    max_length = length
                    new_chain = chain

        if new_chain:
            self.chain = new_chain
            return True

        return False

    def add_to_pending_transactions(self, sender, recipient, amount):
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        }

        if not self.validate_transaction(transaction):
            raise APIException(f"Invalid transaction: {transaction}")

        self.pending_transactions.append(transaction)

        return transaction

    def mine_block_in_network(self, miner):
        last_block = self.chain[-1]
        new_block = self.create_new_block(self.pending_transactions, last_block['hash'])
        new_block['miner'] = miner

        if self.add_block_to_network(new_block):
            self.pending_transactions = []
            return new_block

        return None

    def add_block_to_network(self, new_block):
        if not self.validate_proof(new_block):
            return False

        last_block = self.chain[-1]
        if last_block['index'] + 1 != new_block['index']:
            return False

        if last_block['hash'] != new_block['previous_hash']:
            return False

        self.chain.append(new_block)

        return True

    def add_node_in_network(self, node):
        if node not in self.nodes:
            self.nodes.add(node)

    def mine_pending_transactions_in_network(self, miner):
        self.mine_pending_transactions(miner)

        for node in self.nodes:
            response = requests.get(f"{node}/replace_chain")

            if response.status_code == 200 and response.json()['is_chain_replaced']:
                self.pending_transactions = []
                break

        return None

    def validate_node(self, node):
        response = requests.get(f"{node}/get_chain")

        if response.status_code == 200:
            length = response.json()['length']
            chain = response.json()['chain']

            if length == len(self.chain) and self.validate_chain(chain):
                return True

        return False

    def remove_node(self, node):
        self.nodes.discard(node)

    def validate_block(self, block):
        if block['index'] != self.chain[-1]['index'] + 1:
            return False

        if block['previous_hash'] != self.calculate_hash(self.chain[-1]):
            return False

        if not self.validate_proof(block):
            return False

        return True

    def validate_proof_in_network(self, proof):
        last_block = self.chain[-1]
        new_block = self.create_new_block(self.pending_transactions, last_block['hash'])
        new_block['miner'] = ''
        new_block['proof'] = proof

        return self.add_block_to_network(new_block)

    def get_chain(self):
        return {
            'chain': self.chain,
            'length': len(self.chain)
        }

    def get_chain_in_network(self):
        response = requests.get(f"{self.nodes[0]}/get_chain")

        if response.status_code == 200:
            chain = response.json()['chain']
            length = response.json()['length']

            if length > len(self.chain) and self.validate_chain(chain):
                self.chain = chain

        return {
            'chain': self.chain,
            'length': len(self.chain)
        }

    def replace_chain_in_network(self):
        replaced = False

        for node in self.nodes:
            response = requests.get(f"{node}/get_:
            self.chain = new_chain

@app.route('/mine', methods=['POST'])
def mine():
    miner = request.json['miner']
    blockchain.add_transaction(miner, 'Network', blockchain.mining_reward)
    new_block = blockchain.mine_pending_transactions(miner)
    return jsonify(new_block), 200

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    sender = request.json['sender']
    recipient = request.json['recipient']
    amount = request.json['amount']

    blockchain.add_transaction(sender, recipient, amount)
    return jsonify({'message': 'Transaction added successfully'}), 201

@app.route('/get_chain', methods=['GET'])
def get_chain():
    return jsonify({'chain': blockchain.chain, 'length': len(blockchain.chain)}), 200

@app.route('/validate_chain', methods=['GET'])
def validate_chain():
    if blockchain.validate_chain():
        return jsonify({'message': 'Chain is valid'}), 200
    else:
        return jsonify({'error': 'Chain is not valid'}), 400

if __name__ == '__main__':
    app.run(debug=True)
