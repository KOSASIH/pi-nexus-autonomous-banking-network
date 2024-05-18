import requests


class Binance:
    def __init__(self, api_key=None, api_secret=None):
        self.base_url = "https://api.binance.com"
        self.api_key = api_key
        self.api_secret = api_secret

    def get_ticker_price(self, symbol):
        endpoint = f"/api/v3/ticker/price?symbol={symbol}"
        url = self.base_url + endpoint

        headers = {}

        if self.api_key and self.api_secret:
            headers["X-MBX-APIKEY"] = self.api_key
            timestamp = str(int(round(time.time() * 1000)))
            headers["X-MBX-APIKEY-TIMESTAMP"] = timestamp
            order_string = f"symbol={symbol}&timestamp={timestamp}"
            message = order_string + self.api_secret
            signature = hashlib.sha256(message.encode("utf-8")).hexdigest()
            headers["X-MBX-APIKEY-SIGN"] = signature

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return float(data["price"])
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None


if __name__ == "__main__":
    binance = Binance(api_key="your_api_key", api_secret="your_api_secret")
    btc_usdt_price = binance.get_ticker_price("BTCUSDT")
    print(f"BTC/USDT price: {btc_usdt_price}")
