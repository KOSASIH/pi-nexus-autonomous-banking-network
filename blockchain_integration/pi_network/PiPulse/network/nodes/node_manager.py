import socket
import threading
from node import Node

class NodeManager:
    def __init__(self, node_pool):
        self.node_pool = node_pool
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def remove_node(self, node_id):
        del self.nodes[node_id]

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def start_nodes(self):
        for node in self.nodes.values():
            node.start()

    def stop_nodes(self):
        for node in self.nodes.values():
            node.stop()

    def update_node_utilization(self):
        for node in self.nodes.values():
            node.update_utilization()

    def get_node_info(self, node_id):
        node = self.get_node(node_id)
        if node:
            return node.get_node_info()
        return None

class NodeManagerServer:
    def __init__(self, node_manager, host, port):
        self.node_manager = node_manager
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

    def start(self):
        print(f'Starting node manager server on {self.host}:{self.port}...')
        while True:
            client_socket, address = self.server_socket.accept()
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_thread.start()

    def handle_client(self, client_socket):
        # Handle client requests (e.g., add node, remove node, get node info)
        pass
