import json
import requests
from flask import Flask, jsonify, request
from urllib.parse import urlparse
import threading
import time
import socket

class Node:
    def __init__(self, node_id, port):
        self.node_id = node_id
        self.port = port
        self.peers = set()
        self.transactions = []
        self.blockchain = []  # This should be an instance of your Blockchain class
        self.app = Flask(__name__)
        self.lock = threading.Lock()

        # Set up REST API endpoints
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/transactions/new', methods=['POST'])
        def new_transaction():
            values = request.get_json()
            required = ['sender', 'recipient', 'amount']
            if not all(k in values for k in required):
                return 'Missing values', 400

            transaction = self.create_transaction(values['sender'], values['recipient'], values['amount'])
            response = {'message': 'Transaction will be added to the next block', 'transaction': transaction}
            return jsonify(response), 201

        @self.app.route('/mine', methods=['GET'])
        def mine():
            # Implement mining logic here
            block = self.mine_block()
            response = {'message': 'New block mined', 'block': block}
            return jsonify(response), 200

        @self.app.route('/chain', methods=['GET'])
        def full_chain():
            response = {'chain': self.blockchain, 'length': len(self.blockchain)}
            return jsonify(response), 200

        @self.app.route('/nodes/register', methods=['POST'])
        def register_nodes():
            values = request.get_json()
            nodes = values.get('nodes')
            if nodes is None:
                return 'Error: Please supply a valid list of nodes', 400

            for node in nodes:
                self.add_peer(node)

            response = {'message': 'New nodes have been added', 'total_nodes': list(self.peers)}
            return jsonify(response), 201

        @self.app.route('/nodes/resolve', methods=['GET'])
        def consensus():
            # Implement consensus algorithm here
            response = {'message': 'Consensus algorithm executed'}
            return jsonify(response), 200

    def create_transaction(self, sender, recipient, amount):
        # Create a new transaction and add it to the list
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
            'timestamp': time.time()
        }
        with self.lock:
            self.transactions.append(transaction)
        return transaction

    def mine_block(self):
        # Implement block mining logic
        # This is a placeholder for actual mining logic
        block = {
            'index': len(self.blockchain) + 1,
            'transactions': self.transactions,
            'timestamp': time.time(),
            'previous_hash': self.get_last_block_hash()
        }
        self.blockchain.append(block)
        self.transactions = []  # Reset the transaction list
        return block

    def get_last_block_hash(self):
        return self.blockchain[-1]['hash'] if self.blockchain else '0'

    def add_peer(self, node):
        parsed_url = urlparse(node)
        self.peers.add(parsed_url.netloc)

    def broadcast_transaction(self, transaction):
        for peer in self.peers:
            try:
                url = f'http://{peer}/transactions/new'
                requests.post(url, json=transaction)
            except requests.exceptions.RequestException:
                print(f"Could not connect to peer {peer}")

    def start(self):
        self.app.run(host='0.0.0.0', port=self.port)

if __name__ == '__main__':
    node_id = socket.gethostname()  # Use the hostname as the node ID
    port = 5000  # Change this to your desired port
    node = Node(node_id, port)
    node.start()
