import json
import socket
import threading
from typing import List, Dict, Tuple
from cryptography.fernet import Fernet
from .node import Node

class P2PNetwork:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.node_id = str(uuid4())
        self.config = config
        self.nodes_dir = self.config.NODES_DIR
        self.load_nodes()
        self.generate_keys()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", self.config.NETWORK_PORT))
        self.server_socket.listen()
        self.server_thread = threading.Thread(target=self.handle_incoming_connections)
        self.server_thread.start()

    def load_nodes(self) -> None:
        if not os.path.exists(self.nodes_dir):
            os.makedirs(self.nodes_dir)
        node_ids = os.listdir(self.nodes_dir)
        for node_id in node_ids:
            node_dir = os.path.join(self.nodes_dir, node_id)
            if os.path.isdir(node_dir):
                node_info_path = os.path.join(node_dir, "node_info.json")
                if os.path.exists(node_info_path):
                    with open(node_info_path, "r") as node_info_file:
                        node_info = json.load(node_info_file)
                        node = Node(
                            node_id=node_info["node_id"],
                            host=node_info["host"],
                            port=node_info["port"]
                        )
                        node.load_keys()
                        node.load_node_info()
                        self.nodes[node.address] = node

    def generate_keys(self) -> None:
        self.nodes[self.node_id] = Node(
            node_id=self.node_id,
            host=self.config.HOST,
            port=self.config.PORT
        )
        self.nodes[self.node_id].generate_keys()
        self.nodes[self.node_id].save_keys()
        self.nodes[self.node_id].save_node_info()

    def handle_incoming_connections(self) -> None:
        while True:
            client_socket, client_address = self.server_socket.accept()
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_thread.start()

    def handle_client(self, client_socket: socket.socket, client_address: Tuple[str, int]) -> None:
        try:
            client_node = self.create_node_from_socket(client_socket)
            self.nodes[client_node.address] = client_node
            client_node.save_node_info()
            self.announce_new_node(client_node)
            self.sync_blockchain(client_node)
            self.handle_message_loop(client_socket)
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
            del self.nodes[client_address]

    def create_node_from_socket(self, socket: socket.socket) -> Node:
        node_info_json = socket.recv(1024).decode()
        node_info = json.loads(node_info_json)
        node_id = node_info["node_id"]
        host = node_info["host"]
        port = node_info["port"]
        node = Node(node_id=node_id, host=host, port=port)
        node.public_key = Fernet(node_info["public_key"]).decrypt(node_info["public_key"].encode())
        return node

    def announce_new_node(self, node: Node) -> None:
        for other_node in self.nodes.values():
            if other_node.address != node.address and not node.is_connected_to(other_node):
                node.connect_to_node(other_node)
                other_node.connect_to_node(node)
                node.save_node_info()
                other_node.save_node_info()

    def sync_blockchain(self, node: Node) -> None:
        if node.address in self.nodes:
            blockchain = node.blockchain
            for other_node in self.nodes.values():
                if other_node.address != node.address and other_node.blockchain.chain_height > blockchain.chain_height:
                    blockchain.replace_chain(other_node.blockchain.chain)

    def handle_message_loop(self, socket: socket.socket) -> None:
        while True:
            try:
                encrypted_message = socket.recv(1024)
                if not encrypted_message:
                    break
                message = self.nodes[self.node_id].decrypt_message(encrypted_message)
                message_dict = json.loads(message)
                if "type" in message_dict:
                    if message_dict["type"] == "new_block":
                        block = Block.from_dict(message_dict["data"])
                        self.nodes[self.node_id].blockchain.add_block(block)
                    elif message_dict["type"] == "new_transaction":
                        transaction = Transaction.from_dict(message_dict["data"])
                        self.nodes[self.node_id].blockchain.add_transaction(transaction)
            except Exception as e:
                print(f"Error handling message: {e}")
                break
