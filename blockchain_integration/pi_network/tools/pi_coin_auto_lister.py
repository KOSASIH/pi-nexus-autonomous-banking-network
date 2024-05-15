import os
import json
import requests
from websocket import create_connection
from decimal import Decimal

# Configuration
PI_COIN_SYMBOL = 'PI'
STABLE_VALUE = Decimal('314.159')
EXCHANGES = [
    {'name': 'Binance', 'api_url': 'https://api.binance.com/api/v3', 'ws_url': 'wss://stream.binance.com:9443/stream'},
    {'name': 'Kraken', 'api_url': 'https://api.kraken.com/0/public', 'ws_url': 'wss://ws.kraken.com'},
    {'name': 'Huobi', 'api_url': 'https://api.huobi.pro/market', 'ws_url': 'wss://api.huobi.pro/ws'},
    # Add more exchanges as needed
]

# Function to list PI coin on an exchange
def list_pi_coin_on_exchange(exchange):
    print(f"Listing PI coin on {exchange['name']}...")
    
    # Create a new market pair for PI coin
    market_pair = f"{PI_COIN_SYMBOL}USDT"
    response = requests.post(f"{exchange['api_url']}/ticker/new", json={'symbol': market_pair, 'quoteAsset': 'USDT'})
    if response.status_code != 200:
        print(f"Error creating market pair on {exchange['name']}: {response.text}")
        return
    
    # Set the stable value for PI coin
    response = requests.post(f"{exchange['api_url']}/ticker/update", json={'symbol': market_pair, 'price': STABLE_VALUE})
    if response.status_code != 200:
        print(f"Error setting stable value on {exchange['name']}: {response.text}")
        return
    
    print(f"PI coin listed on {exchange['name']} with stable value of {STABLE_VALUE}!")

# Function to connect to an exchange's WebSocket API
def connect_to_exchange_ws(exchange):
    print(f"Connecting to {exchange['name']} WebSocket API...")
    ws = create_connection(exchange['ws_url'])
    print(f"Connected to {exchange['name']} WebSocket API!")
    return ws

# Function to send a WebSocket message to an exchange
def send_ws_message(ws, message):
    ws.send(json.dumps(message))

# Main function
def main():
    for exchange in EXCHANGES:
        list_pi_coin_on_exchange(exchange)
        ws = connect_to_exchange_ws(exchange)
        send_ws_message(ws, {'method': 'SUBSCRIBE', 'params': [f"{PI_COIN_SYMBOL}@trade"], 'id': 1})
        print(f"Subscribed to {PI_COIN_SYMBOL} trade updates on {exchange['name']}!")

if __name__ == '__main__':
    main()
