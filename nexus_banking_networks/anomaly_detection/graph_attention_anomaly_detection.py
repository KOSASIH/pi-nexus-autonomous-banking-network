import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GraphAttentionLayer

class GraphAttentionAnomalyDetector(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GraphAttentionAnomalyDetector, self).__init__()
        self.conv1 = GraphAttentionLayer(num_features, hidden_dim)
        self.conv2 = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

class GraphData(Data):
    def __init__(self, x, edge_index, batch):
        super(GraphData, self).__init__(x=x, edge_index=edge_index, batch=batch)

def create_graph_data(data, num_nodes):
    edge_index = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index = torch.tensor(edge_index).t().contiguous()
    x = torch.tensor(data.values)
    batch = torch.tensor([0] * num_nodes)
    return GraphData(x, edge_index, batch)

# Example usage
data = pd.read_csv('data.csv')
graph_data = create_graph_data(data, num_nodes=100)
model = GraphAttentionAnomalyDetector(num_features=data.shape[1], hidden_dim=128, output_dim=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(graph_data)
    loss = criterion(output, torch.tensor([0.0]))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Anomaly detection
anomaly_scores = []
for i in range(len(data)):
    graph_data.x = torch.tensor(data.iloc[i:i+100].values)
    output = model(graph_data)
    anomaly_scores.append(output.item())

anomaly_threshold = 3
anomalies = [i for i, score in enumerate(anomaly_scores) if score > anomaly_threshold]
print(f'Anomalies detected: {anomalies}')
