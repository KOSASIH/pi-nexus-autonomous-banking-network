class NodeRegistry:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def remove_node(self, node):
        del self.nodes[node.node_id]


# Example usage
node_registry = NodeRegistry()
