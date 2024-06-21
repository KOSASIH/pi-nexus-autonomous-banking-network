import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class FraudDetector(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super(FraudDetector, self).__init__()
        self.conv_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        for conv in self.conv_layers:
            x = torch.relu(conv(x, edge_index))
        return x

# Load the graph data
data = Data(x=torch.tensor([[1, 2], [3, 4], [5, 6]]), edge_index=torch.tensor([[0, 1], [1, 2], [2, 0]]))

# Create the fraud detector model
model = FraudDetector(num_layers=2, hidden_dim=16)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output, torch.tensor([0, 1, 0]))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Use the trained model to detect fraud
output = model(data.x, data.edge_index)
print(output)
