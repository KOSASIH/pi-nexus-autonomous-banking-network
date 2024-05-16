# pi_coin_lister.py

# Pi Coin Listing Exchanges - 2024
# List of potential exchanges for Pi coin listing

exchanges = [
    {
        "name": "Indodax",
        "location": "Indonesia",
        "ticker": "IDX",
        "website": "https://www.indodax.com",
    },
    {
        "name": "Binance",
        "location": "Global",
        "ticker": "BNB",
        "website": "https://www.binance.com",
    },
    {
        "name": "Coinbase",
        "location": "Global",
        "ticker": "CBSE",
        "website": "https://www.coinbase.com",
    },
    {
        "name": "Kraken",
        "location": "Global",
        "ticker": "KRA",
        "website": "https://www.kraken.com",
    },
    # Add more exchanges here
]

# Pi Coin Listing Details
pi_coin_name = "Pi"
pi_coin_symbol = "PI"
pi_coin_value = 314.159
stablecoin_status = True
listing_date = "1 June 2024"

# Print the Pi coin listing details
print(f"Pi Coin Listing Details:")
print(f"Pi Coin Name: {pi_coin_name}")
print(f"Pi Coin Symbol: {pi_coin_symbol}")
print(f"Pi Coin Value: ${pi_coin_value}")
print(f"Stablecoin Status: {stablecoin_status}")
print(f"Listing Date: {listing_date}")
print("\n")

# Print the list of exchanges
print(f"Pi Coin Listing Exchanges:")
for exchange in exchanges:
    print(f"{exchange['name']} ({exchange['ticker']}) - {exchange['location']}")
    print(f"Website: {exchange['website']}")
    print("\n")
