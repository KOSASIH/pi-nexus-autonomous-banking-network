class Edge:
    def __init__(self, edge_id, node1_id, node2_id, attributes):
        self.edge_id = edge_id
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.attributes = attributes

    def __repr__(self):
        return f"Edge({self.edge_id}, {self.node1_id}, {self.node2_id}, {self.attributes})"
