import socket
import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class P2PNode:
    def __init__(self, host: str, port: int, private_key: str, public_key: str):
        self.host = host
        self.port = port
        self.private_key = serialization.load_pem_private_key(private_key.encode(), password=None, backend=default_backend())
        self.public_key = serialization.load_pem_public_key(public_key.encode(), backend=default_backend())
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, node: str) -> None:
        self.socket.connect((node, self.port))

    def send_message(self, message: str) -> None:
        encrypted_message = self.encrypt_message(message)
        self.socket.sendall(encrypted_message)

    def receive_message(self) -> str:
        encrypted_message = self.socket.recv(1024)
        return self.decrypt_message(encrypted_message)

    def encrypt_message(self, message: str) -> bytes:
        encrypted_message = self.public_key.encrypt(
            message.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_message

    def decrypt_message(self, encrypted_message: bytes) -> str:
        decrypted_message = self.private_key.decrypt(
            encrypted_message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_message.decode()
