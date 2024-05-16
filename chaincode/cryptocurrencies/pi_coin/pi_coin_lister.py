# Import required libraries
import requests
import json
from datetime import datetime
import logging
import time
from ratelimit import limits, sleep_and_retry

# Define constants
PI_COIN_SYMBOL = "PI"
PI_COIN_NAME = "Pi Coin"
LISTING_DATE = datetime(2024, 6, 1)
INITIAL_VALUE = 314.159

# Define exchanges list
EXCHANGES = [
    {"name": "Indodax", "api_url": "https://api.indodax.com/api/v1/ticker", "rate_limit": 10, "time_window": 60},
    {"name": "Binance", "api_url": "https://api.binance.com/api/v3/ticker/price", "rate_limit": 120, "time_window": 60},
    {"name": "Kraken", "api_url": "https://api.kraken.com/0/public/Ticker", "rate_limit": 10, "time_window": 60},
    {"name": "Coinbase", "api_url": "https://api.coinbase.com/v2/prices/spot", "rate_limit": 10, "time_window": 60},
    # Add more exchanges as needed
]

# Define Pi Coin listing
PI_COIN_LISTING = {
    "symbol": PI_COIN_SYMBOL,
    "name": PI_COIN_NAME,
    "listing_date": LISTING_DATE.isoformat(),
    "initial_value": INITIAL_VALUE
}

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Function to fetch exchange data with rate limiting
@sleep_and_retry
@limits(calls=1, period=1)
def fetch_exchange_data(exchange):
    try:
        response = requests.get(exchange["api_url"], timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from {exchange['name']}: {e}")
        return None

# Function to list Pi Coin on exchanges
def list_pi_coin_on_exchanges():
    for exchange in EXCHANGES:
        exchange_data = fetch_exchange_data(exchange)
        if exchange_data:
            # Create Pi Coin listing on exchange
            pi_coin_listing_on_exchange = {
                "exchange": exchange["name"],
                "symbol": PI_COIN_SYMBOL,
                "name": PI_COIN_NAME,
                "listing_date": LISTING_DATE.isoformat(),
                "initial_value": INITIAL_VALUE
            }
            logger.info(f"Listing Pi Coin on {exchange['name']}: {pi_coin_listing_on_exchange}")
        else:
            logger.error(f"Failed to fetch data from {exchange['name']}")

# Main function
def main():
    logger.info("Pi Coin Lister v1.0")
    logger.info("Listing Pi Coin on global exchanges...")
    list_pi_coin_on_exchanges()
    logger.info("Pi Coin listing complete!")

if __name__ == "__main__":
    main()
