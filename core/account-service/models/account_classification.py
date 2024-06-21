import torch
import torch.nn as nn
import torch.optim as optim
from graph_convolutional_nn import GraphConvolutionalNN
from transfer_learning import TransferLearning

class AccountClassification(nn.Module):
  def __init__(self, num_layers, hidden_dim):
    super(AccountClassification, self).__init__()
    self.graph_convolutional_nn = GraphConvolutionalNN(num_layers, hidden_dim)
    self.transfer_learning = TransferLearning(hidden_dim)

  def forward(self, x, edge_index):
    graph_embeddings = self.graph_convolutional_nn(x, edge_index)
    classification_output = self.transfer_learning(graph_embeddings)
    return classification_output

# Load the graph data
data = Data(x=torch.tensor([[1, 2], [3, 4], [5, 6]]), edge_index=torch.tensor([[0, 1], [1, 2], [2, 0]]))

# Create the account classification model
model = AccountClassification(num_layers=2, hidden_dim=16)

# Train the model with transfer learning
transfer_learning = TransferLearning(model, data)
transfer_learning.train()

# Use the trained model to classify accounts
classification_output = model(data.x, data.edge_index)
print(classification_output)
