import asyncio
import hashlib
import json

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


class PINetworkProtocol:
    def __init__(self, node_id, private_key, public_key):
        self.node_id = node_id
        self.private_key = private_key
        self.public_key = public_key
        self.network_topology = {}  # Node ID -> (IP, Port)
        self.message_buffer = asyncio.Queue()

    async def establish_connection(self, node_id, ip, port):
        # Establish a secure connection using TLS 1.3
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://{ip}:{port}/connect",
                headers={"Node-ID": self.node_id},
                data={"public_key": self.public_key},
            ) as resp:
                if resp.status == 200:
                    self.network_topology[node_id] = (ip, port)
                    print(f"Connected to node {node_id} at {ip}:{port}")
                else:
                    print(f"Failed to connect to node {node_id} at {ip}:{port}")

    async def send_message(self, node_id, message):
        # Serialize and encrypt the message using AES-256-GCM
        encrypted_message = self._encrypt_message(message)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://{self.network_topology[node_id][0]}:{self.network_topology[node_id][1]}/message",
                headers={"Node-ID": self.node_id},
                data={"message": encrypted_message},
            ) as resp:
                if resp.status == 200:
                    print(f"Sent message to node {node_id}: {message}")
                else:
                    print(f"Failed to send message to node {node_id}: {message}")

    async def receive_message(self):
        # Decrypt and deserialize incoming messages
        while True:
            message = await self.message_buffer.get()
            decrypted_message = self._decrypt_message(message)
            print(
                f"Received message from node {message['node_id']}: {decrypted_message}"
            )

    def _encrypt_message(self, message):
        # AES-256-GCM encryption
        cipher = Cipher(
            algorithms.AES(self.private_key),
            modes.GCM(self.private_key),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        encrypted_message = encryptor.update(message.encode()) + encryptor.finalize()
        return encrypted_message

    def _decrypt_message(self, message):
        # AES-256-GCM decryption
        cipher = Cipher(
            algorithms.AES(self.private_key),
            modes.GCM(self.private_key),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        decrypted_message = decryptor.update(message) + decryptor.finalize()
        return decrypted_message.decode()

    def start(self):
        # Start the protocol
        asyncio.create_task(self.receive_message())
        print("PI Network protocol started")


# Example usage
node_id = "Node-1234"
private_key = rsa.generate_private_key(
    public_exponent=65537, key_size=2048, backend=default_backend()
)
public_key = private_key.public_key()
pi_network_protocol = PINetworkProtocol(node_id, private_key, public_key)
pi_network_protocol.start()
