# node.py
import socket
import threading

from wallet import Wallet

from blockchain import Blockchain


class Node:
    def __init__(self, wallet: Wallet, blockchain: Blockchain):
        self.wallet = wallet
        self.blockchain = blockchain
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.peers = []

    def join_network(self, host: str, port: int) -> None:
        """
        Join the Pi network by connecting to a node.

        Args:
            host (str): The host of the node to connect to.
            port (int): The port of the node to connect to.
        """
        self.socket.connect((host, port))
        self.peers.append((host, port))

        # Start a thread to listen for incoming messages
        threading.Thread(target=self._listen_for_messages).start()

    def leave_network(self) -> None:
        """
        Leave the Pi network by disconnecting from all peers.
        """
        for peer in self.peers:
            self.socket.close(peer)
        self.peers = []

    def send_message(self, message: str) -> None:
        """
        Send a message to all peers.

        Args:
            message (str): The message to send.
        """
        for peer in self.peers:
            self.socket.sendto(message.encode("utf-8"), peer)

    def _listen_for_messages(self) -> None:
        """
        Listen for incoming messages from peers.
        """
        while True:
            message, peer = self.socket.recvfrom(1024)
            print(f'Received message from {peer}: {message.decode("utf-8")}')
