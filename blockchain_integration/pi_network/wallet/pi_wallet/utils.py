import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data(data, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["close"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


def calculate_portfolio_performance(data):
    portfolio_return = data["return"].mean() * 252
    portfolio_volatility = data["return"].std() * np.sqrt(252)
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


def print_portfolio_performance(portfolio_return, portfolio_volatility, sharpe_ratio):
    print("Portfolio Return: {:.2f}%".format(portfolio_return * 100))
    print("Portfolio Volatility: {:.2f}%".format(portfolio_volatility * 100))
    print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
