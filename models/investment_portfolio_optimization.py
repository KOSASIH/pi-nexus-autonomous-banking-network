import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize

class InvestmentPortfolioOptimizationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(InvestmentPortfolioOptimizationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class InvestmentPortfolioOptimizationDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

def train_investment_portfolio_optimization_model(data, labels, epochs=100, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InvestmentPortfolioOptimizationModel(input_dim=data.shape[1], hidden_dim=128, output_dim=data.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = InvestmentPortfolioOptimizationDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    return model

def optimize_portfolio(model, data, target_return, target_risk):
    def objective(weights):
        portfolio_return = np.dot(weights, data.mean())
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(model(data), weights)))
        return -(portfolio_return - target_return) ** 2 + (portfolio_volatility - target_risk) ** 2

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(data.shape[1])]
    result = minimize(objective, np.ones(data.shape[1]) / data.shape[1], constraints=constraints, bounds=bounds)
    return result.x
