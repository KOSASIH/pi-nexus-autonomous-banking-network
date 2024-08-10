import networkx as nx
from graph import Graph
from visualization import Visualization

class Topology:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = graph.nodes
        self.edges = graph.edges

    def add_node(self, node_id, node_type, **kwargs):
        self.graph.add_node(node_id, node_type, **kwargs)

    def add_edge(self, node1_id, node2_id, **kwargs):
        self.graph.add_edge(node1_id, node2_id, **kwargs)

    def remove_node(self, node_id):
        self.graph.remove_node(node_id)

    def remove_edge(self, node1_id, node2_id):
        self.graph.remove_edge(node1_id, node2_id)

    def get_node(self, node_id):
        return self.graph.get_node(node_id)

    def get_edge(self, node1_id, node2_id):
        return self.graph.get_edge(node1_id, node2_id)

    def get_shortest_path(self, node1_id, node2_id):
        return nx.shortest_path(self.graph, node1_id, node2_id)

    def visualize(self):
        visualization = Visualization(self.graph)
        visualization.show()

class HierarchicalTopology(Topology):
    def __init__(self, graph):
        super().__init__(graph)
        self.clusters = {}

    def add_cluster(self, cluster_id, nodes):
        self.clusters[cluster_id] = nodes

    def get_cluster(self, cluster_id):
        return self.clusters.get(cluster_id)

class MeshTopology(Topology):
    def __init__(self, graph):
        super().__init__(graph)
        self.dimensions = (0, 0)

    def set_dimensions(self, dimensions):
        self.dimensions = dimensions

    def get_dimensions(self):
        return self.dimensions
