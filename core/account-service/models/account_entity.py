import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import TemporalGraphEmbedding
from explainers import Explainer

class AccountEntity(nn.Module):
  def __init__(self, num_layers, hidden_dim):
    super(AccountEntity, self).__init__()
    self.temporal_embedding = TemporalGraphEmbedding(num_layers, hidden_dim)

  def forward(self, x, edge_index, timestamps):
    embeddings = self.temporal_embedding(x, edge_index, timestamps)
    return embeddings

# Load the temporal graph data
data = Data(x=torch.tensor([[1, 2], [3, 4], [5, 6]]), edge_index=torch.tensor([[0, 1], [1, 2], [2, 0]]), timestamps=torch.tensor([1, 2, 3]))

# Create the account entity model
model = AccountEntity(num_layers=2, hidden_dim=16)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
  optimizer.zero_grad()
  embeddings = model(data.x, data.edge_index, data.timestamps)
  loss = criterion(embeddings, torch.tensor([0, 1, 0]))
  loss.backward()
  optimizer.step()
  print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Use the trained model to embed accounts
embeddings = model(data.x, data.edge_index, data.timestamps)
print(embeddings)

# Explain the model's predictions using SHAP values
explainer = Explainer(model, data)
shap_values = explainer.shap_values(data.x, data.edge_index, data.timestamps)
print(shap_values)
