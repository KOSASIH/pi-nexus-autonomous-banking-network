# ai-module/model.py
import torch
import torch.nn as nn
import torch.optim as optim


class PiNexusAIModel(nn.Module):
    def __init__(self):
        super(PiNexusAIModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)  # input layer (128) -> hidden layer (64)
        self.fc2 = nn.Linear(64, 32)  # hidden layer (64) -> hidden layer (32)
        self.fc3 = nn.Linear(32, 1)  # hidden layer (32) -> output layer (1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = PiNexusAIModel()
