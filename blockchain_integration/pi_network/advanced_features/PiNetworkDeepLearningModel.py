# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Class for deep learning model
class PiNetworkDeepLearningModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # Function to train the model
    def train(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = nn.MSELoss()(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # Function to evaluate the model
    def evaluate(self, inputs):
        outputs = self.model(inputs)
        return outputs

# Example usage
model = PiNetworkDeepLearningModel(input_dim=10, hidden_dim=20, output_dim=5)
inputs = torch.randn(1, 10)
labels = torch.randn(1, 5)
loss = model.train(inputs, labels)
print(f"Loss: {loss:.4f}")
