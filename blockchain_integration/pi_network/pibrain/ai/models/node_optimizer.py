import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class NodeOptimizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NodeOptimizer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NodeOptimizerDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return {
            'x': torch.tensor(x, dtype=torch.float),
            'y': torch.tensor(y, dtype=torch.float)
        }

def train_node_optimizer(model, dataset, batch_size, epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in data_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    return model

def load_node_optimizer_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(['node_performance'], axis=1).values
    y = data['node_performance'].values
    return X, y

# Example usage:
X, y = load_node_optimizer_data('node_performance_data.csv')
dataset = NodeOptimizerDataset(X, y)
model = NodeOptimizer(input_dim=X.shape[1], hidden_dim=128, output_dim=1)
model = train_node_optimizer(model, dataset, batch_size=32, epochs=100, learning_rate=0.001)
