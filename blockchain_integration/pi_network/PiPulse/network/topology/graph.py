import networkx as nx

class Graph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node_id, node_type, **kwargs):
        self.graph.add_node(node_id, node_type=node_type, **kwargs)

    def add_edge(self, node1_id, node2_id, **kwargs):
        self.graph.add_edge(node1_id, node2_id, **kwargs)

    def remove_node(self, node_id):
        self.graph.remove_node(node_id)

    def remove_edge(self, node1_id, node2_id):
        self.graph.remove_edge(node1_id, node2_id)

    def get_node(self, node_id):
        return self.graph.nodes.get(node_id)

    def get_edge(self, node1_id, node2_id):
        return self.graph.get_edge_data(node1_id, node2_id)

    def nodes(self):
        return list(self.graph.nodes)

    def edges(self):
        return list(self.graph.edges)
