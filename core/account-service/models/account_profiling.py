import torch
import torch.nn as nn
import torch.optim as optim
from transformers import TransformerModel
from multi_task_learning import MultiTaskLearning

class AccountProfiling(nn.Module):
  def __init__(self, num_layers, hidden_dim):
    super(AccountProfiling, self).__init__()
    self.transformer_model = TransformerModel(num_layers, hidden_dim)
    self.multi_task_learning = MultiTaskLearning(hidden_dim)

  def forward(self, x):
    transformer_output = self.transformer_model(x)
    profiling_output = self.multi_task_learning(transformer_output)
    return profiling_output

# Load the account data
data = torch.tensor([[1, 2], [3, 4], [5, 6]])

# Create the account profiling model
model = AccountProfiling(num_layers=2, hidden_dim=16)

# Train the model with multi-task learning
multi_task_learning = MultiTaskLearning(model, data)
multi_task_learning.train()

# Use the trained model to profile accounts
profiling_output = model(data)
print(profiling_output)
