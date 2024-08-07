import socket
import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class IBCConnector:
    def __init__(self, host: str, port: int, private_key: str, public_key: str):
        self.host = host
        self.port = port
        self.private_key = serialization.load_pem_private_key(private_key.encode(), password=None, backend=default_backend())
        self.public_key = serialization.load_pem_public_key(public_key.encode(), backend=default_backend())

    def establish_connection(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        return sock

    def send_message(self, message: str, sock: socket.socket) -> None:
        encrypted_message = self.encrypt_message(message)
        sock.sendall(encrypted_message)

    def receive_message(self, sock: socket.socket) -> str:
        encrypted_message = sock.recv(1024)
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
