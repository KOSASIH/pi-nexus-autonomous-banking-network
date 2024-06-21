import torch
import torch.nn as nn
import torch.optim as optim
from bayesian_neural_networks import BayesianNeuralNetwork
from uncertainty_quantification import UncertaintyQuantification

class AccountRiskAssessment(nn.Module):
  def __init__(self, num_layers, hidden_dim):
    super(AccountRiskAssessment, self).__init__()
    self.bayesian_neural_network = BayesianNeuralNetwork(num_layers, hidden_dim)

  def forward(self, x):
    risk_score = self.bayesian_neural_network(x)
    return risk_score

# Load the account data
data = torch.tensor([[1, 2], [3, 4], [5, 6]])

# Create the account risk assessment model
model = AccountRiskAssessment(num_layers=2, hidden_dim=16)

# Train the model with Bayesian inference
bayesian_inference = BayesianInference(model, data)
bayesian_inference.train()

# Use the trained model to assess account risk
risk_score = model(data)
print(risk_score)

# Quantify the uncertainty of the risk score
uncertainty_quantification = UncertaintyQuantification(model, data)
uncertainty = uncertainty_quantification.quantify_uncertainty()
print(uncertainty)
