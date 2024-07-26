# sidra_chain_neural_network.py
import torch
import torch.nn as nn
import torch.optim as optim

class SidraChainNeuralNetwork(nn.Module):
    def __init__(self):
        super(SidraChainNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SidraChainNeuralNetworkTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, dataset, batch_size=32, epochs=10):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def evaluate(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        return accuracy
