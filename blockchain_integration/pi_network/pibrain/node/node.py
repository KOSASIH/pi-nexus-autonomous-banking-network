# node.py

import os
import sys
import logging
import threading
import socket
import json
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

class Node:
    """Node class."""

    def __init__(self, node_id: str, node_name: str, node_type: str, node_address: str, node_port: int):
        self.node_id = node_id
        self.node_name = node_name
        self.node_type = node_type
        self.node_address = node_address
        self.node_port = node_port
        self.config = NodeConfig(node_id, node_name, node_type, node_address, node_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((node_address, node_port))
        self.socket.listen(5)
        self.clients = {}
        self.threads = {}
        self.shutdown = False

    def start(self) -> None:
        """Start the node."""
        _LOGGER.info(f'Starting node {self.node_id}...')
        self.config.load_config(f'configs/{self.node_id}.json')
        self.socket.listen(5)
        _LOGGER.info(f'Node {self.node_id} started.')

    def stop(self) -> None:
        """Stop the node."""
        _LOGGER.info(f'Stopping node {self.node_id}...')
        self.shutdown = True
        self.socket.close()
        _LOGGER.info(f'Node {self.node_id} stopped.')

    def handle_client(self, client_socket: socket.socket, client_address: str) -> None:
        """Handle a client connection."""
        _LOGGER.info(f'Client connected: {client_address}')
        client_id = f'client-{len(self.clients)}'
        self.clients[client_id] = client_socket
        thread = threading.Thread(target=self.handle_client_thread, args=(client_id, client_socket))
        self.threads[client_id] = thread
        thread.start()

    def handle_client_thread(self, client_id: str, client_socket: socket.socket) -> None:
        """Handle a client connection in a separate thread."""
        while not self.shutdown:
            try:
                data = client_socket.recv(1024)
                if data:
                    _LOGGER.info(f'Received data from {client_id}: {data}')
                    self.process_data(client_id, data)
                else:
                    _LOGGER.info(f'Client {client_id} disconnected.')
                    break
            except Exception as e:
                _LOGGER.error(f'Error handling client {client_id}: {e}')
                break
        self.clients.pop(client_id)
        self.threads.pop(client_id)

    def process_data(self, client_id: str, data: bytes) -> None:
        """Process data received from a client."""
        try:
            data_json = json.loads(data.decode('utf-8'))
            _LOGGER.info(f'Received JSON data from {client_id}: {data_json}')
            # Process the data here
            response = {'result': 'success'}
            self.send_response(client_id, response)
        except Exception as e:
            _LOGGER.error(f'Error processing data from {client_id}: {e}')

    def send_response(self, client_id: str, response: Dict[str, Any]) -> None:
        """Send a response to a client."""
        try:
            response_json = json.dumps(response)
            self.clients[client_id].sendall(response_json.encode('utf-8'))
        except Exception as e:
            _LOGGER.error(f'Error sending response to {client_id}: {e}')

def main():
    logging.basicConfig(level=logging.INFO)
    node_id = 'node-1'
    node_name = 'Node 1'
    node_type = 'server'
    node_address = 'localhost'
    node_port = 50051
    node = Node(node_id, node_name, node_type, node_address, node_port)
    node.start()

    try:
        while True:
            client_socket, client_address = node.socket.accept()
            node.handle_client(client_socket, client_address)
    except KeyboardInterrupt:
        node.stop()

if __name__ == '__main__':
    main()
