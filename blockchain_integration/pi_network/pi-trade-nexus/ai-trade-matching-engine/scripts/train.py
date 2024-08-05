import os
import argparse
from models.trade_matching_model import TradeMatchingModel
from models.market_prediction_model import MarketPredictionModel
from data_loader import load_trade_data, load_market_data

def train_trade_matching_model(args):
    # Load trade data
    trade_data = load_trade_data(args.trade_data_path)

    # Create and train trade matching model
    model = TradeMatchingModel()
    model.train(trade_data.drop(["trade_id", "timestamp"], axis=1), trade_data["trade_id"])

    # Save model to file
    model.save(args.model_path)

def train_market_prediction_model(args):
    # Load market data
    market_data = load_market_data(args.market_data_path)

    # Create and train market prediction model
    model = MarketPredictionModel()
    model.train(market_data.drop(["asset_id", "timestamp"], axis=1), market_data["price"])

    # Save model to file
    model.save(args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--trade_data_path", type=str, required=True, help="Path to trade data CSV file")
    parser.add_argument("--market_data_path", type=str, required=True, help="Path to market data CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--model_type", type=str, required=True, choices=["trade_matching", "market_prediction"], help="Type of model to train")

    args = parser.parse_args()

    if args.model_type == "trade_matching":
        train_trade_matching_model(args)
    elif args.model_type == "market_prediction":
        train_market_prediction_model(args)
