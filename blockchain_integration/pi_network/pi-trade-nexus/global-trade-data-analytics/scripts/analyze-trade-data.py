import pandas as pd
from utils.data_loader import load_trade_data
from utils.feature_engineering import extract_trade_features
from models.trade_model import TradeModel

def analyze_trade_data(file_path):
    # Load trade data
    trade_data = load_trade_data(file_path)

    # Extract features from trade data
    trade_features = extract_trade_features(trade_data)

    # Train trade model
    trade_model = TradeModel()
    trade_model.train(trade_features)

    # Evaluate trade model
    trade_mse = trade_model.evaluate(trade_features)
    print(f'Trade Model MSE: {trade_mse:.2f}')

if __name__ == '__main__':
    analyze_trade_data('data/trade-data.csv')
