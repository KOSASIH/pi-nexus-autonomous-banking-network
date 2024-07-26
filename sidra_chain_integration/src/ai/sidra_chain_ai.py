# sidra_chain_ai.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SidraChainAIDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

class SidraChainAIModel(nn.Module):
    def __init__(self):
        super(SidraChainAIModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SidraChainAI:
    def __init__(self):
        self.model = SidraChainAIModel()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, dataset, batch_size=32, epochs=10):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
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
