import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class FraudDetectionModel(nn.Module):
    def __init__(self):
        super(FraudDetectionModel, self).__init__()
        self.conv1 = GCNConv(10, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, edge_index)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = FraudDetectionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Load graph data
graph_data = Data(x=torch.tensor([[...]]), edge_index=torch.tensor([[...]]))

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    output = model(graph_data.x, graph_data.edge_index)
    loss = criterion(output, torch.tensor([1]))  # 1 for fraud, 0 for legitimate
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Test the model
test_output = model(graph_data.x, graph_data.edge_index)
print("Fraud detection accuracy:", torch.sum(test_output.argmax(dim=1) == torch.tensor([1])).item() / len(test_output))
