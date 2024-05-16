# Import required libraries
import requests
import json
from datetime import datetime
import time
import logging
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
PI_COIN_LISTED_DATE = datetime(2024, 6, 1)
PI_COIN_INITIAL_VALUE = 314.159

# Define exchanges list
exchanges: List[Dict[str, str]] = [
    {
        "name": "Indodax",
        "api_url": "https://api.indodax.com/api/v1/ticker",
        "api_rate_limit": 10,
        "listing_url": "https://indodax.com/api/new-market",
        "listing_params": {
            "market_name": "PI_IDR",
            "minimal_amount": 1,
            "minimal_price": 1,
            "maximal_amount": 1000000,
            "maximal_price": 1000000,
            "fee_buyer": 0.002,
            "fee_seller": 0.002,
            "is_active": True,
            "is_hidden": False,
            "is_withdrawal_enabled": True,
            "is_deposit_enabled": True,
            "is_trading_enabled": True,
            "is_market_maker_enabled": True,
            "is_market_taker_enabled": True,
            "is_stop_limit_enabled": True,
            "is_stop_market_enabled": True,
            "is_limit_enabled": True,
            "is_market_enabled": True,
            "is_cancel_order_enabled": True,
            "is_order_book_enabled": True,
            "is_trade_history_enabled": True,
            "is_order_history_enabled": True,
            "is_withdrawal_history_enabled": True,
            "is_deposit_history_enabled": True,
            "is_transfer_enabled": True,
            "is_transfer_history_enabled": True,
            "is_api_enabled": True,
            "is_web_enabled": True,
            "is_mobile_enabled": True,
            "is_internal_enabled": True,
            "is_withdrawal_fee_percentage": False,
            "withdrawal_fee_value": 100,
            "deposit_fee_value": 0,
            "minimal_deposit_value": 1,
            "maximal_deposit_value": 1000000,
            "minimal_withdrawal_value": 1,
            "maximal_withdrawal_value": 1000000,
            "minimal_trade_value": 1,
            "maximal_trade_value": 1000000,
            "minimal_order_value": 1,
            "maximal_order_value": 1000000,
            "minimal_stop_limit_value": 1,
            "maximal_stop_limit_value": 1000000,
            "minimal_stop_market_value": 1,
            "maximal_stop_market_value": 1000000,
            "minimal_limit_value": 1,
            "maximal_limit_value": 1000000,
            "minimal_market_value": 1,
            "maximal_market_value": 1000000,
            "minimal_cancel_order_value": 1,
            "maximal_cancel_order_value": 1000000,
            "minimal_order_book_value": 1,
            "maximal_order_book_value": 1000000,
            "minimal_trade_history_value": 1,
            "maximal_trade_history_value": 1000000,
            "minimal_order_history_value": 1,
            "maximal_order_history_value": 1000000,
            "minimal_withdrawal_history_value": 1,
            "maximal_withdrawal_history_value": 1000000,
            "minimal_deposit_history_value": 1,
            "maximal_deposit_history_value": 1000000,
            "minimal_transfer_value": 1,
            "maximal_transfer_value": 1000000,
            "minimal_transfer_history_value": 1,
            "maximal_transfer_history_value": 1000000,
            "minimal_api_value": 1,
            "maximal_api_value": 1000000,
            "minimal_web_value": 1,
            "maximal_web_value": 1000000,
            "minimal_mobile_value": 1,
            "maximal_mobile_value": 1000000,
            "minimal_internal_value": 1,
            "maximal_internal_value": 1000000,
        },
    },
    # Add more exchanges as needed
]

# Define PI Coin data structure
pi_coin: Dict[str, any] = {
    "name": "PI Coin",
    "symbol": "PIC",
    "listed_date": PI_COIN_LISTED_DATE,
    "initial_value": PI_COIN_INITIAL_VALUE,
    "exchanges": exchanges
}

# Function to fetch current prices from exchanges
def fetch_prices(exchange: Dict[str, str]) -> str:
    try:
        response = requests.get(exchange["api_url"])
        response.raise_for_status()  # Raise an exception for bad status codes
        data = json.loads(response.content)
        return data["price"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching price from {exchange['name']}: {e}")
        return "N/A"

# Function to list PI Coin on exchanges
def list_pi_coin(pi_coin: Dict[str, any]) -> None:
    logger.info("Listing PI Coin on exchanges:")
    for exchange in pi_coin["exchanges"]:
        price = fetch_prices(exchange)
        logger.info(f"  {exchange['name']}: {price}")
        if exchange["name"] == "Indodax":
            list_on_indodax(exchange["listing_url"], exchange["listing_params"])
        time.sleep(60 / exchange["api_rate_limit"])  # Rate limit delay

# Function to list PI Coin on Indodax
def list_on_indodax(listing_url: str, listing_params: Dict[str, any]) -> None:
    try:
        response = requests.post(listing_url, json=listing_params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = json.loads(response.content)
        logger.info(f"    Listed on Indodax: {data['message']}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error listing on Indodax: {e}")

# Main function
def main() -> None:
    logger.info("PI Coin Auto Lister")
    logger.info("--------------------")
    logger.info(f"PI Coin listed on {PI_COIN_LISTED_DATE.strftime('%Y-%m-%d')}")
    logger.info(f"Initial value: ${PI_COIN_INITIAL_VALUE:.2f}")
    list_pi_coin(pi_coin)

if __name__ == "__main__":
    main()
