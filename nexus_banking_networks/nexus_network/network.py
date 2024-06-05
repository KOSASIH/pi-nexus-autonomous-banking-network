import networkx as nx
from.node import Node
from.edge import Edge

class NexusNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, node_type, attributes):
        node = Node(node_id, node_type, attributes)
        self.nodes[node_id] = node
        self.graph.add_node(node_id)

    def add_edge(self, edge_id, node1_id, node2_id, attributes):
        edge = Edge(edge_id, node1_id, node2_id, attributes)
        self.edges[edge_id] = edge
        self.graph.add_edge(node1_id, node2_id)

    def get_nodes(self):
        return list(self.nodes.values())

    def get_edges(self):
        return list(self.edges.values())

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_edge(self, edge_id):
        return self.edges.get(edge_id)
