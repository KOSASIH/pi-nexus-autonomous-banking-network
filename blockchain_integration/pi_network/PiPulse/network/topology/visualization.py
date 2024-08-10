import matplotlib.pyplot as plt
import networkx as nx

class Visualization:
    def __init__(self, graph):
        self.graph = graph

    def show(self):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.show()
