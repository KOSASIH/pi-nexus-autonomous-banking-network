# agi.py
import torch
import torch.nn as nn
import torch.optim as optim

class EonixAGI:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, data):
        inputs, labels = data
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_model(self, data):
        inputs, labels = data
        inputs = torch.tensor(inputs)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / len(labels)
        return accuracy
