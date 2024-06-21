import torch
import torch.nn as nn
import torch.optim as optim
from graphsage import GraphSAGE
from reinforcement_learning import ReinforcementLearning

class AccountRecommender(nn.Module):
  def __init__(self, num_layers, hidden_dim):
    super(AccountRecommender, self).__init__()
    self.graphsage = GraphSAGE(num_layers, hidden_dim)
    self.reinforcement_learning = ReinforcementLearning(hidden_dim)

  def forward(self, x, edge_index):
    graphsage_output = self.graphsage(x, edge_index)
    recommendation_output = self.reinforcement_learning(graphsage_output)
    return recommendation_output

# Load the graph data
data = Data(x=torch.tensor([[1, 2], [3, 4], [5, 6]]), edge_index=torch.tensor([[0, 1], [1, 2], [2, 0]]))

# Create the account recommender model
model = AccountRecommender(num_layers=2, hidden_dim=16)

# Train the model with reinforcement learning
reinforcement_learning = ReinforcementLearning(model, data)
reinforcement_learning.train()

# Use the trained model to recommend accounts
recommendation_output = model(data.x, data.edge_index)
print(recommendation_output)
