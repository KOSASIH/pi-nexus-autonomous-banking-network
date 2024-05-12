from cryptography.fernet import Fernet
from typing import List, Dict, Tuple
from uuid import uuid4
from . import config

class Node:
    def __init__(self, node_id: str, host: str, port: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.public_key = None
        self.private_key = Fernet.generate_key()
        self.outbound_connections: List[str] = []
        self.inbound_connections: List[str] = []

    def connect_to_node(self, node: "Node") -> None:
        if node.address not in self.outbound_connections:
            self.outbound_connections.append(node.address)
            node.inbound_connections.append(self.address)

    def disconnect_from_node(self, node: "Node") -> None:
        if node.address in self.outbound_connections:
            self.outbound_connections.remove(node.address)
            node.inbound_connections.remove(self.address)

    def send_message(self, message: str, node: "Node") -> None:
        encrypted_message = self.encrypt_message(message)
        node.receive_message(encrypted_message)

    def receive_message(self, encrypted_message: bytes) -> None:
        message = self.decrypt_message(encrypted_message)
        print(f"Received message: {message}")

    def encrypt_message(self, message: str) -> bytes:
        message_bytes = message.encode()
        encrypted_message = Fernet(self.public_key).encrypt(message_bytes)
        return encrypted_message

    def decrypt_message(self, encrypted_message: bytes) -> str:
        message_bytes = Fernet(self.private_key).decrypt(encrypted_message)
        message = message_bytes.decode()
        return message

    def get_peers(self) -> Tuple[List[str], List[str]]:
        return self.outbound_connections, self.inbound_connections

    def is_connected_to(self, node: "Node") -> bool:
        return node.address in self.outbound_connections or node.address in self.inbound_connections
def generate_keys(self) -> None:
        self.public_key, self.private_key = Fernet.generate_key_pair()

    def save_keys(self) -> None:
        with open(f"{config.NODES_DIR}/{self.node_id}/keys.pem", "wb") as keys_file:
            keys_file.write(self.public_key + self.private_key)

    def load_keys(self) -> None:
        with open(f"{config.NODES_DIR}/{self.node_id}/keys.pem", "rb") as keys_file:
            keys = keys_file.read()
            self.public_key = keys[:32]
            self.private_key = keys[32:]

    def save_node_info(self) -> None:
        node_info = {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "public_key": self.public_key,
            "outbound_connections": self.outbound_connections,
            "inbound_connections": self.inbound_connections
        }
        with open(f"{config.NODES_DIR}/{self.node_id}/node_info.json", "w") as node_info_file:
            node_info_file.write(json.dumps(node_info))

    def load_node_info(self) -> None:
        with open(f"{config.NODES_DIR}/{self.node_id}/node_info.json", "r") as node_info_file:
            node_info = json.load(node_info_file)
            self.node_id = node_info["node_id"]
            self.host = node_info["host"]
            self.port = node_info["port"]
            self.public_key = node_info["public_key"]
            self.outbound_connections = node_info["outbound_connections"]
            self.inbound_connections = node_info["inbound_connections"]
