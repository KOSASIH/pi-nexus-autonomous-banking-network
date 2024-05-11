import random
import time

class NetworkManagement:
    def __init__(self, nodes):
        self.nodes = nodes

    def add_node(self, node):
        self.nodes.append(node)
        self.update_topology()

    def remove_node(self, node):
        self.nodes.remove(node)
        self.update_topology()

    def update_topology(self):
        # update network topology based on current nodes
        pass

    def simulate_traffic(self):
        # simulate network traffic for a period of time
        for _ in range(10):
            for node in self.nodes:
                node.receive_data(random.randint(1, 100))
            time.sleep(1)
