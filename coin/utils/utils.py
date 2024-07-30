import requests
import json

def get_coin_data(symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
    response = requests.get(url)
    data = json.loads(response.text)
    return data

def get_coin_price(symbol):
    data = get_coin_data(symbol)
    return data["market_data"]["current_price"]["usd"]

def get_coin_market_cap(symbol):
    data = get_coin_data(symbol)
    return data["market_data"]["market_cap"]["usd"]

def get_coin_volume(symbol):
    data = get_coin_data(symbol)
    return data["market_data"]["total_volume"]["usd"]

def get_coin_info(symbol):
    data = get_coin_data(symbol)
    return {
        "name": data["name"],
        "symbol": data["symbol"],
        "price": data["market_data"]["current_price"]["usd"],
        "market_cap": data["market_data"]["market_cap"]["usd"],
        "volume": data["market_data"]["total_volume"]["usd"]
  }
