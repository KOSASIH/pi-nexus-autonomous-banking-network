import requests
import json
from datetime import datetime

# Configuration
EXCHANGES_API = "https://api.exchanges.com/v1/exchanges"
COIN_API = "https://api.coins.com/v1/coins"
PI_COIN_SYMBOL = "PI"
TARGET_DATE = datetime(2024, 6, 1)
TARGET_VALUE = 314.159

def get_exchanges():
    """Get a list of global exchanges"""
    response = requests.get(EXCHANGES_API)
    return json.loads(response.content)

def get_coin_list(exchange):
    """Get a list of coins for a given exchange"""
    response = requests.get(f"{COIN_API}/{exchange}/coins")
    return json.loads(response.content)

def filter_pi_coins(coins):
    """Filter coins with PI symbol and target value"""
    return [coin for coin in coins if coin["symbol"] == PI_COIN_SYMBOL and coin["price"] == TARGET_VALUE]

def main():
    exchanges = get_exchanges()
    pi_coins = []

    for exchange in exchanges:
        coins = get_coin_list(exchange["id"])
        pi_coins.extend(filter_pi_coins(coins))

    print("Automatic PI Coin Lister - 1 June 2024")
    print("
