import torch
import torch.nn as nn
import torch.optim as optim
from bayesian_neural_networks import BayesianNeuralNetwork
from uncertainty_quantification import UncertaintyQuantification

class AccountRiskAssessment(nn.Module):
  def __init__(self, num_layers, hidden_dim):
    super(AccountRiskAssessment, self).__init__()
    self.bayesian_neural_network = BayesianNeuralNetwork(num_layers, hidden_dim)
    self.uncertainty_quantification = UncertaintyQuantification(hidden_dim)

  def forward(self, x):
    risk_score = self.bayesian_neural_network(x)
    uncertainty = self.uncertainty_quantification(risk_score)
    return risk_score, uncertainty

# Load the account data
data = torch.tensor([[1, 2], [3, 4], [5, 6]])

# Create the account risk assessment model
model = AccountRiskAssessment(num_layers=2, hidden_dim=16)

# Train the model with Bayesian inference
bayesian_inference = BayesianInference(model, data)
bayesian_inference.train()

# Use the trained model to assess account risk
risk_score, uncertainty = model(data)
print(risk_score, uncertainty)
