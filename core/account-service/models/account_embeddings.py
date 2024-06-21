import torch
import torch.nn as nn
import torch.optim as optim
from hyperbolic_nn import HyperbolicNN
from attention import AttentionMechanism

class AccountEmbeddings(nn.Module):
  def __init__(self, num_layers, hidden_dim):
    super(AccountEmbeddings, self).__init__()
    self.hyperbolic_nn = HyperbolicNN(num_layers, hidden_dim)
    self.attention_mechanism = AttentionMechanism(hidden_dim)

  def forward(self, x):
    hyperbolic_embeddings = self.hyperbolic_nn(x)
    attention_weights = self.attention_mechanism(hyperbolic_embeddings)
    return attention_weights * hyperbolic_embeddings

# Load the account data
data = torch.tensor([[1, 2], [3, 4], [5, 6]])

# Create the account embeddings model
model = AccountEmbeddings(num_layers=2, hidden_dim=16)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
  optimizer.zero_grad()
  embeddings = model(data)
  loss = criterion(embeddings, torch.tensor([0, 1, 0]))
  loss.backward()
  optimizer.step()
  print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Use the trained model to embed accounts
embeddings = model(data)
print(embeddings)
