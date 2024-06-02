import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_annualized_standard_deviation(data):
    """
    Calculate the annualized standard deviation of a portfolio.

    Parameters:
    data (pd.DataFrame): Portfolio data with daily returns.

    Returns:
    float: Annualized standard deviation.
    """
    std_dev = data.std() * np.sqrt(253)
    return std_dev

def calculate_annualized_variance(std_dev):
    """
    Calculate the annualized variance of a portfolio.

    Parameters:
    std_dev (float): Annualized standard deviation.

    Returns:
    float: Annualized variance.
    """
    var = std_dev ** 2
    return var

def calculate_sharpe_ratio(data, risk_free_rate):
    """
    Calculate the Sharpe ratio of a portfolio.

    Parameters:
    data (pd.DataFrame): Portfolio data with daily returns.
    risk_free_rate (float): Risk-free rate.

    Returns:
    float: Sharpe ratio.
    """
    mean_return = data.mean() * 253
    std_dev = calculate_annualized_standard_deviation(data)
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratio

def plot_annualized_standard_deviation(data):
    """
    Plot the annualized standard deviation of a portfolio over time.

    Parameters:
    data (pd.DataFrame): Portfolio data with daily returns.
    """
    std_dev = calculate_annualized_standard_deviation(data)
    plt.plot(std_dev.index, std_dev.values)
    plt.xlabel('Year')
    plt.ylabel('Annualized Standard Deviation')
    plt.title('Annualized Standard Deviation Over Time')
    plt.show()

def plot_sharpe_ratio(data, risk_free_rate):
    """
    Plot the Sharpe ratio of a portfolio over time.

    Parameters:
    data (pd.DataFrame): Portfolio data with daily returns.
    risk_free_rate (float): Risk-free rate.
    """
    sharpe_ratio = calculate_sharpe_ratio(data, risk_free_rate)
    plt.plot(sharpe_ratio.index, sharpe_ratio.values)
    plt.xlabel('Year')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Over Time')
    plt.show()

# Load data
data = pd.read_csv('portfolio_data.csv', index_col='Date', parse_dates=['Date'])

# Calculate annualized standard deviation
std_dev = calculate_annualized_standard_deviation(data)

# Calculate annualized variance
var = calculate_annualized_variance(std_dev)

# Calculate Sharpe ratio
risk_free_rate = 0.02
sharpe_ratio = calculate_sharpe_ratio(data, risk_free_rate)

# Plot annualized standard deviation
plot_annualized_standard_deviation(data)

# Plot Sharpe ratio
plot_sharpe_ratio(data, risk_free_rate)
