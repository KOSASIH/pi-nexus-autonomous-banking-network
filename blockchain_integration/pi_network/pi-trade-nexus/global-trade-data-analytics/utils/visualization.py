import matplotlib.pyplot as plt
import seaborn as sns

def visualize_trade_data(trade_data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='date', y='value', data=trade_data)
    plt.title('Trade Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()

def visualize_market_data(market_data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='date', y='close', data=market_data)
    plt.title('Market Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.show()
