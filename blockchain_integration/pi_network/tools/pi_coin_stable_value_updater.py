import json
import os
from decimal import Decimal

import requests

# Configuration
PI_COIN_SYMBOL = "PI"
STABLE_VALUE = Decimal("314.159")
EXCHANGES = [
    {"name": "Binance", "api_url": "https://api.binance.com/api/v3"},
    {"name": "Kraken", "api_url": "https://api.kraken.com/0/public"},
    {"name": "Huobi", "api_url": "https://api.huobi.pro/market"},
    # Add more exchanges as needed
]


# Function to update the stable value of PI coin on an exchange
def update_pi_coin_stable_value_on_exchange(exchange):
    print(f"Updating stable value of PI coin on {exchange['name']}...")

    # Get the current price of PI coin on the exchange
    response = requests.get(
        f"{exchange['api_url']}/ticker/price", params={"symbol": PI_COIN_SYMBOL}
    )
    if response.status_code != 200:
        print(f"Error getting current price on {exchange['name']}: {response.text}")
        return

    current_price = Decimal(response.json()["price"])

    # Calculate the difference between the current price and the stable value
    price_diff = STABLE_VALUE - current_price

    # If the difference is significant, update the stable value
    if abs(price_diff) > Decimal("0.01"):
        response = requests.post(
            f"{exchange['api_url']}/ticker/update",
            json={"symbol": PI_COIN_SYMBOL, "price": STABLE_VALUE},
        )
        if response.status_code != 200:
            print(f"Error updating stable value on {exchange['name']}: {response.text}")
            return

        print(
            f"Updated stable value of PI coin on {exchange['name']} to {STABLE_VALUE}!"
        )


# Main function
def main():
    while True:
        for exchange in EXCHANGES:
            update_pi_coin_stable_value_on_exchange(exchange)
        time.sleep(60)  # Update every 1 minute


if __name__ == "__main__":
    main()
