import socket
import threading
from cryptography import generate_public_key, generate_private_key, sign_transaction, verify_signature
from blockchain import Blockchain, Transaction
from services.analytics import analyze_transactions

class Consensus:
    def __init__(self, blockchain, host, port):
        self.blockchain = blockchain
        self.host = host
        self.port = port
        self.peers = []
        self.peer_sockets = []

        # Start the peer discovery thread
        self.discovery_thread = threading.Thread(target=self.discover_peers)
        self.discovery_thread.start()

        # Start the consensus thread
        self.consensus_thread = threading.Thread(target=self.reach_consensus)
        self.consensus_thread.start()

    def discover_peers(self):
        # TODO: implement logic to discover peers
        pass

    def connect_to_peer(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        self.peer_sockets.append(sock)

    def send_blockchain(self, sock):
        blockchain_json = self.blockchain.to_json()
        sock.sendall(blockchain_json.encode('utf-8'))

    def receive_blockchain(self, sock):
        blockchain_json = sock.recv(1024).decode('utf-8')
        new_blockchain = Blockchain.from_json(blockchain_json)

        if self.blockchain.chain[-1].index < new_blockchain.chain[-1].index:
            self.blockchain = new_blockchain

    def request_new_chain(self):
        for sock in self.peer_sockets:
            self.send_blockchain(sock)

        for sock in self.peer_sockets:
            self.receive_blockchain(sock)

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

    def reach_consensus(self):
        while True:
            if not self.blockchain.is_valid():
                raise Exception("Blockchain is invalid")

            if len(self.blockchain.chain) < 2:
                continue

            if self.blockchain.chain[-1].index % 10 == 0:
                self.blockchain.mine_pending_transactions()
                analyze_transactions(self.blockchain)

            self.request_new_chain()

            if len(self.blockchain.chain) < 2:
                continue

            if not self.is_new_chain_valid(self.blockchain.chain):
                self.blockchain.chain = self.blockchain.create_genesis_block()

            for sock in self.peer_sockets:
                self.send_blockchain(sock)

            time.sleep(1)

    def add_transaction(self, transaction):
        self.blockchain.add_transaction(transaction)

    def sign_transaction(self, transaction, private_key):
        transaction.sign_transaction(private_key)

    def verify_transaction(self, transaction, public_key):
        return verify_signature(transaction.to_json(), transaction.signature, public_key)

    def create_transaction(self, sender_id, receiver_id, amount):
        public_key = self.get_public_key(sender_id)
        private_key = self.get_private_key(sender_id)

        transaction = Transaction(sender_id, receiver_id, amount)
        self.sign_transaction(transaction, private_key)

        return transaction

    def get_public_key(self, user_id):
        # TODO: implement logic to retrieve public key for a user
        pass

    def get_private_key(self, user_id):
        # TODO: implement logic to retrieve private key for a user
        pass
