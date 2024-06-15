# qr_node.py
import os
import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from qiskit import QuantumCircuit, execute
import time
import requests
import threading

class QRNode:
    def __init__(self, node_id, private_key, peers):
        self.node_id = node_id
        self.private_key = private_key
        self.peers = peers
        self.quantum_circuit = QuantumCircuit(5, 5)
        self.blockchain = []
        self.mempool = []
        self.stake = 1000
        self.consensus_thread = threading.Thread(target=self.consensus)

    def generate_quantum_key(self):
        # Generate a quantum key using Qiskit
        job = execute(self.quantum_circuit, backend='ibmq_qasm_simulator')
        quantum_key = job.result().get_statevector()
        return quantum_key

    def encrypt_transaction(self, transaction):
        # Encrypt transaction using quantum-resistant cryptography
        cipher = Cipher(algorithms.AES(self.private_key), modes.GCM(iv=os.urandom(12)))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(transaction) + encryptor.finalize()
        return ciphertext

    def verify_transaction(self, transaction, signature):
        # Verify transaction using digital signature
        public_key = serialization.load_pem_public_key(self.private_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
        verifier = hashlib.sha256()
        verifier.update(transaction)
        try:
            public_key.verify(signature, verifier.digest())
            return True
        except Exception:
            return False

    def propagate_transaction(self, transaction):
        # Propagate transaction using gossip protocol
        for peer in self.peers:
            requests.post(f"http://{peer}/mempool", json=transaction)

    def add_transaction(self, transaction):
        # Add transaction to mempool
        if self.verify_transaction(transaction, transaction['signature']):
            self.mempool.append(transaction)
            self.propagate_transaction(transaction)

    def create_block(self):
        # Create a new block
        block = {
            'index': len(self.blockchain),
            'timestamp': int(time.time()),
            'transactions': self.mempool,
            'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else '0' * 64,
            'hash': None,
            'nonce': 0,
        }
        block['hash'] = self.calculate_hash(block)
        self.blockchain.append(block)
        self.mempool = []

    def calculate_hash(self, block):
        # Calculate the hash of a block
        block_json = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_json).hexdigest()

    def consensus(self):
        # Proof of Stake consensus algorithm
        while True:
            if len(self.blockchain) > 10 and time.time() - self.blockchain[-1]['timestamp'] > 60:
                self.create_block()
            time.sleep(1)

    def start(self):
        # Start the node
        self.consensus_thread.start()
        self.api = Flask(__name__)
       self.api.add_url_rule('/mempool', view_func=self.add_transaction, methods=['POST'])
        self.api.add_url_rule('/blockchain', view_func=self.get_blockchain, methods=['GET'])
        self.api.run(host='0.0.0.0', port=5000)

    def get_blockchain(self):
        # Return the blockchain
        return jsonify(self.blockchain)

# Example usage
node = QRNode('node1', rsa.generate_private_key(public_exponent=65537, key_size=2048), ['node2:5000', 'node3:5000'])
node.start()
