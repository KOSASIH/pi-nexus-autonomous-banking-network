import pandas as pd
from utils.data_loader import load_market_data
from utils.visualization import visualize_market_data

def visualize_market_data(file_path):
    # Load market data
    market_data = load_market_data(file_path)

    # Visualize market data
    visualize_market_data(market_data)

if __name__ == '__main__':
    visualize_market_data('data/market-data.csv')
