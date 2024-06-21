import torch
import torch.nn as nn
import torch.optim as optim
from deep_embedded_clustering import DeepEmbeddedClustering
from self_supervised_learning import SelfSupervisedLearning

class AccountClustering(nn.Module):
  def __init__(self, num_layers, hidden_dim):
    super(AccountClustering, self).__init__()
    self.deep_embedded_clustering = DeepEmbeddedClustering(num_layers, hidden_dim)
    self.self_supervised_learning = SelfSupervisedLearning(hidden_dim)

  def forward(self, x):
    embedded_clusters = self.deep_embedded_clustering(x)
    self_supervised_loss = self.self_supervised_learning(embedded_clusters)
    return embedded_clusters, self_supervised_loss

# Load the account data
data = torch.tensor([[1, 2], [3, 4], [5, 6]])

# Create the account clustering model
model = AccountClustering(num_layers=2, hidden_dim=16)

# Train the model with self-supervised learning
self_supervised_learning = SelfSupervisedLearning(model, data)
self_supervised_learning.train()

# Use thetrained model to cluster accounts
embedded_clusters, _ = model(data)
print(embedded_clusters)
