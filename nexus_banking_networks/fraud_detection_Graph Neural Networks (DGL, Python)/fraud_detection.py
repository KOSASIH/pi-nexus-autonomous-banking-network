# fraud_detection.py
import dgl
import torch
import torch.nn as nn
import torch.optim as optim

class FraudDetection(nn.Module):
  def __init__(self):
    super(FraudDetection, self).__init__()
    self.gcn = dgl.nn.GraphConv(128, 64)
    self.fc = nn.Linear(64, 2)

  def forward(self, g, h):
    h = self.gcn(g, h)
    h = torch.relu(h)
    h = self.fc(h)
    return h

# Example usage:
model = FraudDetection()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

graph = # load transaction graph data
node_features = # load node features
labels = # load fraud labels

for epoch in range(10):
  optimizer.zero_grad()
  outputs = model(graph, node_features)
  loss = criterion(outputs, labels)
  loss.backward()
  optimizer.step()
  print("Epoch:", epoch, "Loss:", loss.item())
