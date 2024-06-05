class Node:

    def __init__(self, node_id, node_type, attributes):
        self.node_id = node_id
        self.node_type = node_type
        self.attributes = attributes

    def __repr__(self):
        return f"Node({self.node_id}, {self.node_type}, {self.attributes})"
