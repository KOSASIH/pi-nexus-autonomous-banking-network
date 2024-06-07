import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GenerativeAdversarialNetworkPortfolioOptimizer:
    def __init__(self, assets, returns, cov_matrix):
        self.assets = assets
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.generator = Generator(input_dim=returns.shape[1], hidden_dim=128, output_dim=returns.shape[1])
        self.discriminator = Discriminator(input_dim=returns.shape[1], hidden_dim=128, output_dim=1)

    def optimize_portfolio(self, num_epochs=100):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.01)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.01)
        for epoch in range(num_epochs):
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            z = torch.randn(1, self.returns.shape[1])
            x_fake = self.generator(z)
            x_real = torch.tensor(self.returns)
            y_fake = self.discriminator(x_fake)
            y_real = self.discriminator(x_real)
            loss_g = -torch.mean(y_fake)
            loss_d = torch.mean(y_fake) - torch.mean(y_real)
            loss_g.backward()
            loss_d.backward()
            optimizer_g.step()
            optimizer_d.step()
            print(f'Epoch {epoch+1}, Loss G: {loss_g.item()}, Loss D: {loss_d.item()}')

        optimal_weights = self.generator(z).detach().numpy()
        return optimal_weights

# Example usage
assets = ['Asset1', 'Asset 2', 'Asset 3']
returns = np.array([0.03, 0.05, 0.01])
cov_matrix = np.array([[0.001, 0.002, 0.001], [0.002, 0.004, 0.002], [0.001, 0.002, 0.003]])
portfolio_optimizer = GenerativeAdversarialNetworkPortfolioOptimizer(assets, returns, cov_matrix)
optimal_weights = portfolio_optimizer.optimize_portfolio()
print(f'Optimal weights: {optimal_weights}')
