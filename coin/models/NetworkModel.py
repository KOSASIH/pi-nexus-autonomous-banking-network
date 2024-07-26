from typing import List

class NetworkModel:
    def __init__(self):
        self.nodes = []

    def add_node(self, node: 'NodeModel') -> None:
        # Add a new node to the network
        self.nodes.append(node)

    def remove_node(self, node: 'NodeModel') -> None:
        # Remove a node from the network
        self.nodes.remove(node)

    def connect_nodes(self, node1: 'NodeModel', node2: 'NodeModel') -> None:
        # Connect two nodes
        node1.connect_peer(node2)
        node2.connect_peer(node1)

    def disconnect_nodes(self, node1: 'NodeModel', node2: 'NodeModel') -> None:
        # Disconnect two nodes
        node1.disconnect_peer(node2)
        node2.disconnect_peer(node1)

    def sync_network(self) -> None:
        # Synchronize the entire network
        for node in self.nodes:
            node.sync_blockchain()
