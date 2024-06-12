import qaoa

# Define a QAOA optimizer
def qaoa_optimizer(problem, num_qubits, num_layers):
    optimizer = qaoa.QAOAOptimizer(problem, num_qubits, num_layers)
    return optimizer

# Use QAOA to optimize a portfolio
def optimize_portfolio(optimizer, portfolio_data):
    optimized_portfolio = optimizer.optimize(portfolio_data)
    return optimized_portfolio
