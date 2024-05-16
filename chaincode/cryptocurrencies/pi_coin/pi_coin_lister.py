import requests
import json
import pandas as pd
from datetime import datetime
from pytz import timezone
import os
import threading
from queue import Queue
from sqlalchemy import create_engine
import ccxt

# Set API endpoints and credentials
EXCHANGE_API_ENDPOINTS = {
    'Binance': 'https://api.binance.com',
    'Coinbase': 'https://api.coinbase.com',
    'Kraken': 'https://api.kraken.com',
    'Indodax': 'https://api.indodax.com'
}
API_CREDENTIALS = {
    'Binance': {'api_key': os.environ['BINANCE_API_KEY'], 'api_secret': os.environ['BINANCE_API_SECRET']},
    'Coinbase': {'api_key': os.environ['COINBASE_API_KEY'], 'api_secret': os.environ['COINBASE_API_SECRET']},
    'Kraken': {'api_key': os.environ['KRAKEN_API_KEY'], 'api_secret': os.environ['KRAKEN_API_SECRET']}
}

# Set PI Coin details
PI_COIN_SYMBOL = 'PI'
LISTING_DATE = datetime(2024, 6, 1, tzinfo=timezone('UTC'))

# Set database connection
DB_ENGINE = create_engine('postgresql://user:password@host:port/dbname')

# Function to fetch exchange data
def fetch_exchange_data(exchange, queue):
    try:
        if exchange == 'Indodax':
            exchange_obj = ccxt.indodax()
            data = exchange_obj.fetch_ticker('PI_IDR')
        else:
            endpoint = EXCHANGE_API_ENDPOINTS[exchange]
            credentials = API_CREDENTIALS[exchange]
            headers = {'X-MBX-APIKEY': credentials['api_key'], 'X-MBX-SECRET-KEY': credentials['api_secret']}
            response = requests.get(endpoint, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
        queue.put((exchange, data))
    except requests.exceptions.RequestException as e:
        print(f'Error fetching data from {exchange}: {e}')
        queue.put((exchange, None))

# Function to extract PI Coin data from exchange data
def extract_pi_coin_data(exchange_data, exchange):
    pi_coin_data = {}
    if exchange == 'Indodax':
        pi_coin_data[exchange] = {
            'symbol': 'PI_IDR',
            'price': exchange_data['last'],
            'volume': exchange_data['volume'],
            'listing_date': LISTING_DATE
        }
    else:
        for symbol_info in exchange_data['symbols']:
            if symbol_info['baseId'] == PI_COIN_SYMBOL:
                pi_coin_data[exchange] = {
                    'symbol': symbol_info['symbol'],
                    'price': symbol_info['price'],
                    'volume': symbol_info['volume'],
                    'listing_date': LISTING_DATE
                }
                break
    return pi_coin_data

# Function to list PI Coin on global exchanges
def list_pi_coin_on_exchanges(queue):
    pi_coin_list = []
    threads = []
    for exchange in EXCHANGE_API_ENDPOINTS:
        t = threading.Thread(target=fetch_exchange_data, args=(exchange, queue))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    while not queue.empty():
        exchange, data = queue.get()
        if data:
            pi_coin_data = extract_pi_coin_data(data, exchange)
            if pi_coin_data:
                pi_coin_list.append(pi_coin_data)
    return pi_coin_list
