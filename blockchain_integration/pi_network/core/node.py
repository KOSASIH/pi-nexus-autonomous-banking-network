# Pi Network Node
import socket
import threading
from pi_network.core.consensus_engine import ConsensusEngine

class Node:
    def __init__(self, node_id, network_id, host, port):
        self.node_id = node_id
        self.network_id = network_id
        self.host = host
        self.port = port
        self.consensus_engine = ConsensusEngine(network_id, node_id)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(5)

    def start(self):
        # Start node and listen for incoming connections
        print(f"Node {self.node_id} started on {self.host}:{self.port}")
        while True:
            conn, addr = self.socket.accept()
            threading.Thread(target=self.handle_connection, args=(conn, addr)).start()

    def handle_connection(self, conn, addr):
        # Handle incoming connection and process messages
        while True:
            message = conn.recv(1024)
            if not message:
                break
            self.process_message(message)

    def process_message(self, message):
        # Process message and update blockchain if necessary
        if message.startswith(b"BLOCK"):
            block = Block.from_bytes(message[5:])
            if self.consensus_engine.add_block(block):
                print(f"Added block {block.hash} to blockchain")
        elif message.startswith(b"TX"):
            tx = Transaction.from_bytes(message[2:])
            # Process transaction and update blockchain if necessary
            pass
