# Import required libraries and frameworks
import json
import os
import threading
from datetime import datetime
from queue import Queue

import pandas as pd
import requests
from pytz import timezone
from sqlalchemy import create_engine

# Set API endpoints and credentials
EXCHANGE_API_ENDPOINTS = {
    "Binance": "https://api.binance.com/api/v3/exchangeInfo",
    "Coinbase": "https://api.coinbase.com/v2/exchange-rates",
    "Kraken": "https://api.kraken.com/0/public/Ticker",
}
API_CREDENTIALS = {
    "Binance": {
        "api_key": os.environ["BINANCE_API_KEY"],
        "api_secret": os.environ["BINANCE_API_SECRET"],
    },
    "Coinbase": {
        "api_key": os.environ["COINBASE_API_KEY"],
        "api_secret": os.environ["COINBASE_API_SECRET"],
    },
    "Kraken": {
        "api_key": os.environ["KRAKEN_API_KEY"],
        "api_secret": os.environ["KRAKEN_API_SECRET"],
    },
}

# Set PI Coin details
PI_COIN_SYMBOL = "PI"
LISTING_DATE = datetime(2024, 6, 1, tzinfo=timezone("UTC"))

# Set database connection
DB_ENGINE = create_engine("postgresql://user:password@host:port/dbname")


# Function to fetch exchange data
def fetch_exchange_data(exchange, queue):
    try:
        endpoint = EXCHANGE_API_ENDPOINTS[exchange]
        credentials = API_CREDENTIALS[exchange]
        headers = {
            "X-MBX-APIKEY": credentials["api_key"],
            "X-MBX-SECRET-KEY": credentials["api_secret"],
        }
        response = requests.get(endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        queue.put((exchange, response.json()))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {exchange}: {e}")
        queue.put((exchange, None))


# Function to extract PI Coin data from exchange data
def extract_pi_coin_data(exchange_data):
    pi_coin_data = {}
    for symbol, info in exchange_data["symbols"].items():
        if symbol == PI_COIN_SYMBOL:
            pi_coin_data[exchange] = {
                "symbol": symbol,
                "price": info["price"],
                "volume": info["volume"],
                "listing_date": LISTING_DATE,
            }
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
            pi_coin_data = extract_pi_coin_data(data)
            if pi_coin_data:
                pi_coin_list.append(pi_coin_data)
    return pi_coin_list


# Function to store PI Coin data in database
def store_pi_coin_data(pi_coin_list):
    df = pd.DataFrame(pi_coin_list)
    df.to_sql("pi_coin_list", con=DB_ENGINE, if_exists="replace", index=False)


# Main function
def main():
    queue = Queue()
    pi_coin_list = list_pi_coin_on_exchanges(queue)
    if pi_coin_list:
        print("PI Coin listed on the following exchanges:")
        for exchange, pi_coin_data in pi_coin_list.items():
            print(
                f'{exchange}: {pi_coin_data["symbol"]} - Price: {pi_coin_data["price"]} - Volume: {pi_coin_data["volume"]} - Listing Date: {pi_coin_data["listing_date"]}'
            )
        store_pi_coin_data(pi_coin_list)
    else:
        print("PI Coin not listed on any exchanges.")


if __name__ == "__main__":
    main()
