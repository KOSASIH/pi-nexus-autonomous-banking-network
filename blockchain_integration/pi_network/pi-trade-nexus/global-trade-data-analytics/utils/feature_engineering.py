import pandas as pd

def extract_features(trade_data, market_data):
    # Extract relevant features from trade data
    trade_features = pd.DataFrame({
        'date': trade_data['date'],
        'country': trade_data['country'],
        'product': trade_data['product'],
        'quantity': trade_data['quantity'],
        'value': trade_data['value']
    })

    # Extract relevant features from market data
    market_features = pd.DataFrame({
        'date': market_data['date'],
        'arket_index': market_data['market_index'],
        'open': market_data['open'],
        'high': market_data['high'],
        'low': market_data['low'],
        'close': market_data['close']
    })

    # Merge trade and market features
    features = pd.merge(trade_features, market_features, on='date')

    return features
