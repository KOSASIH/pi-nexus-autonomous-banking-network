import socket
import pickle
from blockchain_interface import BlockchainInterface

class DistributedLedger:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.blockchain_interface = BlockchainInterface()
        self.nodes = []

    def start(self):
        while True:
            connection, address = self.socket.accept()
            self.nodes.append(connection)
            self.handle_connection(connection)

    def handle_connection(self, connection):
        while True:
            message = self.receive_message(connection)
            if message:
                if message['type'] == 'get_chain':
                    self.send_chain(connection)
                elif message['type'] == 'get_pending_transactions':
                    self.send_pending_transactions(connection)
                elif message['type'] == 'transaction':
                    self.add_transaction(message['transaction'])
                    self.broadcast_transaction(message['transaction'])
                elif message['type'] == 'block':
                    self.add_block(message['block'])
                    self.broadcast_block(message['block'])
                elif message['type'] == 'get_nodes':
                    self.send_nodes(connection)

    def send_message(self, connection, message):
        connection.sendall(pickle.dumps(message))

    def receive_message(self, connection):
        data = connection.recv(1024)
        if not data:
            return None
        return pickle.loads(data)

    def send_chain(self, connection):
        self.send_message(connection, {'type': 'chain', 'chain': self.blockchain_interface.chain})

    def send_pending_transactions(self, connection):
        self.send_message(connection, {'type': 'pending_transactions', 'pending_transactions': self.blockchain_interface.pending_transactions})

    def add_transaction(self, transaction):
        self.blockchain_interface.add_transaction(transaction)

    def broadcast_transaction(self, transaction):
        for node in self.nodes:
            self.send_message(node, {'type': 'transaction', 'transaction': transaction})

    def add_block(self, block):
        self.blockchain_interface.add_block(block)

    def broadcast_block(self, block):
        for node in self.nodes:
            self.send_message(node, {'type': 'block', 'block': block})

    def send_nodes(self, connection):
        self.send_message(connection, {'type': 'nodes', 'nodes': self.nodes})

if __name__ == '__main__':
    distributed_ledger = DistributedLedger('localhost', 8080)
    distributed_ledger.start()
