import asyncio
import json

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


class NodeManager:
    def __init__(self, node_registry):
        self.node_registry = node_registry
        self.node_creation_queue = asyncio.Queue()
        self.node_update_queue = asyncio.Queue()
        self.node_deletion_queue = asyncio.Queue()

    async def create_node(self, node_id, ip, port, public_key):
        # Create a new node and add it to the registry
        node = Node(node_id, ip, port, public_key)
        self.node_registry.add_node(node)
        print(f"Node {node_id} created at {ip}:{port}")
        return node

    async def update_node(self, node_id, ip, port, public_key):
        # Update an existing node in the registry
        node = self.node_registry.get_node(node_id)
        if node:
            node.ip = ip
            node.port = port
            node.public_key = public_key
            print(f"Node {node_id} updated to {ip}:{port}")
        else:
            print(f"Node {node_id} not found")

    async def delete_node(self, node_id):
        # Remove a nodefrom the registry
        node = self.node_registry.get_node(node_id)
        if node:
            self.node_registry.remove_node(node)
            print(f"Node {node_id} deleted")
        else:
            print(f"Node {node_id} not found")

    async def process_node_creation(self):
        while True:
            node_data = await self.node_creation_queue.get()
            await self.create_node(
                node_data["node_id"],
                node_data["ip"],
                node_data["port"],
                node_data["public_key"],
            )

    async def process_node_update(self):
        while True:
            node_data = await self.node_update_queue.get()
            await self.update_node(
                node_data["node_id"],
                node_data["ip"],
                node_data["port"],
                node_data["public_key"],
            )

    async def process_node_deletion(self):
        while True:
            node_id = await self.node_deletion_queue.get()
            await self.delete_node(node_id)

    async def start(self):
        # Start the node manager
        asyncio.create_task(self.process_node_creation())
        asyncio.create_task(self.process_node_update())
        asyncio.create_task(self.process_node_deletion())
        print("Node manager started")


# Example usage
node_registry = NodeRegistry()
node_manager = NodeManager(node_registry)
node_manager.start()
