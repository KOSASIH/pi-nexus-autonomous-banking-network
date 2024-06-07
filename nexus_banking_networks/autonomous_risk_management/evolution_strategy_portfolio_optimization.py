import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

class EvolutionStrategyPortfolioOptimizer:
    def __init__(self, assets, returns, cov_matrix):
        self.assets = assets
        self.returns = returns
        self.cov_matrix = cov_matrix

    def portfolio_return(self, weights):
        return np.sum(self.returns * weights)

    def portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def optimize_portfolio(self, num_generations=100, pop_size=50):
        bounds = [(0, 1) for _ in range(len(self.assets))]
        result = differential_evolution(self.portfolio_volatility, bounds, args=(self.returns,), popsize=pop_size, maxiter=num_generations)
        return result.x

    def optimize_portfolio_with_constraints(self, num_generations=100, pop_size=50):
        bounds = [(0, 1) for _ in range(len(self.assets))]
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = differential_evolution(self.portfolio_volatility, bounds, args=(self.returns,), popsize=pop_size, maxiter=num_generations, constraints=constraints)
        return result.x

# Example usage
assets = ['Asset 1', 'Asset 2', 'Asset 3']
returns = np.array([0.03, 0.05, 0.01])
cov_matrix = np.array([[0.001, 0.002, 0.001], [0.002, 0.004, 0.002], [0.001, 0.002, 0.003]])
portfolio_optimizer = EvolutionStrategyPortfolioOptimizer(assets, returns, cov_matrix)
optimal_weights = portfolio_optimizer.optimize_portfolio()
print(f'Optimal weights: {optimal_weights}')
