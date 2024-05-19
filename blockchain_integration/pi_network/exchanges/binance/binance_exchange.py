import requests
from typing import Dict, Union

class BinanceExchange:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"

    def _get_signed_request_headers(self, endpoint: str) -> Dict[str, str]:
        """Returns the signed request headers for a given endpoint."""
        timestamp = str(int(requests.get("https://api.binance.com/api/v3/time").json()["serverTime"] / 1000))
        signature = self._create_signature(endpoint, timestamp)
        return {
            "X-MBX-APIKEY": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
        }

    def _create_signature(self, endpoint: str, timestamp: str) -> str:
        """Creates a signature for a given endpoint and timestamp."""
        query_string = "{}&timestamp={}".format(endpoint, timestamp)
        secret_key = self.api_secret.encode("utf-8")
        return requests.utils.requote_uri(query_string).encode("utf-8").hex()[:64].upper()

    def fetch_exchange_rates(self) -> Dict[str, float]:
        """Fetches the current exchange rates for Binance."""
        url = f"{self.base_url}/api/v3/ticker/price"
        response = requests.get(url)
        exchange_rates = {}
        for rate in response.json():
            exchange_rates[rate["symbol"].replace("USDT", "")] = float(rate["price"])
        return exchange_rates

    def place_order(self, order: Dict[str, Union[str, float]]) -> Dict[str, Union[str, float]]:
        """Places an order on Binance."""
        url = f"{self.base_url}/api/v3/order"
        headers = self._get_signed_request_headers(url)
        response = requests.post(url, json=order, headers=headers)
        return response.json()

    def get_account_balance(self) -> Dict[str, float]:
        """Fetches the current account balance for Binance."""
        url = f"{self.base_url}/api/v3/account"
        headers = self._get_signed_request_headers(url)
        response = requests.get(url, headers=headers)
        balance = {}
        for asset in response.json()["balances"]:
            if float(asset["free"]) > 0.0:
                balance[asset["asset"]] = float(asset["free"])
        return balance
