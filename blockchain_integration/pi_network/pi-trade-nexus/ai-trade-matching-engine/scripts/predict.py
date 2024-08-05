import os
import argparse
from models.trade_matching_model import TradeMatchingModel
from models.market_prediction_model import MarketPredictionModel
from data_loader import load_trade_data, load_market_data

def predict_trade_matching(args):
    # Load trade data
    trade_data = load_trade_data(args.trade_data_path)

    # Load trained trade matching model
    model = TradeMatchingModel.load(args.model_path)

    # Make predictions
    predictions = model.predict(trade_data.drop(["trade_id", "timestamp"], axis=1))

    # Save predictions to file
    pd.DataFrame(predictions, columns=["predicted_trade_id"]).to_csv(args.output_path, index=False)

def predict_market_prediction(args):
    # Load market data
    market_data = load_market_data(args.market_data_path)

    # Load trained market prediction model
    model = MarketPredictionModel.load(args.model_path)

    # Make predictions
    predictions = model.predict(market_data.drop(["asset_id", "timestamp"], axis=1))

    # Save predictions to file
    pd.DataFrame(predictions, columns=["predicted_price"]).to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions")
    parser.add_argument("--trade_data_path", type=str, required=True, help="Path to trade data CSV file")
    parser.add_argument("--market_data_path", type=str, required=True, help="Path to market data CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--model_type", type=str, required=True, choices=["trade_matching", "market_prediction"], help="Type of model to use for prediction")

    args = parser.parse_args()

    if args.model_type == "trade_matching":
        predict_trade_matching(args)
    elif args.model_type == "market_prediction":
        predict_market_prediction(args)
