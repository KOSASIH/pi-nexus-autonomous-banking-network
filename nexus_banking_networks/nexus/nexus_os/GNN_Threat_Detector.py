import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class GNNThreatDetector:
    def __init__(self, num_nodes, num_edges, hidden_size):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.GraphConv(hidden_size, hidden_size),
            nn.GraphConv(hidden_size, hidden_size)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, dataset):
        for epoch in range(100):
            for graph, label in dataset:
                graph = torch.tensor(graph)
                label = torch.tensor(label)
                self.optimizer.zero_grad()
                output = self.model(graph)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

    def detect_threat(self, graph):
        graph = torch.tensor(graph)
        output = self.model(graph)
        return output

# Example usage:
gnn_threat_detector = GNNThreatDetector(100, 200, 128)
dataset = [(np.random.rand(100, 100), np.random.randint(0, 2, size=(100,))) for _ in range(100)]
gnn_threat_detector.train_model(dataset)

# Detect a threat
graph = np.random.rand(100, 100)
threat_detection = gnn_threat_detector.detect_threat(graph)
print(f'Threat detection: {threat_detection}')
