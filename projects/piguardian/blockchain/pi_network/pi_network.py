class PiNetwork:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def get_nodes(self):
        return self.nodes

    def get_node_by_id(self, node_id):
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_node_by_public_key(self, public_key):
        for node in self.nodes:
            if node.public_key == public_key:
                return node
        return None

    def add_node_to_network(self, node):
        self.add_node(node)
        for existing_node in self.nodes:
            existing_node.add_peer(node)
            node.add_peer(existing_node)

    def remove_node_from_network(self, node):
        self.nodes.remove(node)
        for existing_node in self.nodes:
            existing_node.peers.remove(node)
            node.peers.remove(existing_node)

    def broadcast_transaction(self, transaction):
        for node in self.nodes:
            node.add_transaction(transaction)

    def broadcast_block(self, block):
        for node in self.nodes:
            node.add_block(block)

    def resolve_conflicts(self):
        longest_chain = []
        for node in self.nodes:
            if len(node.blockchain) > len(longest_chain):
                longest_chain = node.blockchain
        for node in self.nodes:
            node.blockchain = longest_chain
