import socket
import threading

class MinerNode:
    def __init__(self, node_id, blockchain):
        self.node_id = node_id
        self.blockchain = blockchain
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        # Start the miner node
        self.socket.bind(("localhost", 8080 + self.node_id))
        self.socket.listen(5)
        print(f"Miner node {self.node_id} started")

        # Start a new thread to listen for incoming connections
        threading.Thread(target=self.listen_for_connections).start()

    def listen_for_connections(self):
        while True:
            # Accept incoming connections
            conn, addr = self.socket.accept()
            print(f"Connected by {addr}")

            # Handle incoming messages
            threading.Thread(target=self.handle_message, args=(conn,)).start()

    def handle_message(self, conn):
        # Handle incoming messages (e.g. new block announcements)
        pass
