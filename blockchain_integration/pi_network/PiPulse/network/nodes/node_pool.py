import os
import torch

class NodePool:
    def __init__(self, node_type, num_nodes, gpu_ids):
        self.node_type = node_type
        self.num_nodes = num_nodes
        self.gpu_ids = gpu_ids
        self.nodes = []

    def create_nodes(self):
        for i in range(self.num_nodes):
            node = self.node_type(f'node-{i}', gpu_id=self.gpu_ids[i % len(self.gpu_ids)])
            self.nodes.append(node)

    def get_node(self, node_id):
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_available_node(self):
        for node in self.nodes:
            if node.cpu_util < 50 and node.memory_util < 50:
                return node
        return None

    def get_node_info(self):
        node_info = []
        for node in self.nodes:
            node_info.append(node.get_node_info())
        return node_info

if __name__ == '__main__':
    gpu_ids = [0, 1, 2, 3]
    node_pool = NodePool(WorkerNode, 8, gpu_ids)
    node_pool.create_nodes()
    print(node_pool.get_node_info())
