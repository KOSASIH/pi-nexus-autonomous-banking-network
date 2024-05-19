import requests
from typing import Dict, Union

class CoinbaseExchange:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coinbase.com"

    def _get_signed_request_headers(self, endpoint: str) -> Dict[str, str]:
        """Returns the signed request headers for a given endpoint."""
        timestamp = str(int(requests.get("https://api.coinbase.com/v2/time").json()["seconds"]))
        message = f"{timestamp}{endpoint}"
        secret_key = self.api_secret.encode("utf-8")
        signature = requests.hmac.new(secret_key, message.encode("utf-8"), hashlib.sha256).hexdigest()
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
        }

    def fetch_exchange_rates(self) -> Dict[str, float]:
        """Fetches the current exchange rates for Coinbase."""
        url = f"{self.base_url}/v2/prices"
        response = requests.get(url)
        exchange_rates = {}
        for rate in response.json()["data"]:
            if rate["currency"] == "USD":
                exchange_rates[rate["base"]] = float(rate["amount"])
        return exchange_rates

    def place_order(self, order: Dict[str, Union[str, float]]) -> Dict[str, Union[str, float]]:
        """Places an order on Coinbase."""
        url = f"{self.base_url}/v2/orders"
        headers = self._get_signed_request_headers(url)
        response = requests.post(url, json=order, headers=headers)
        return response.json()

    def get_account_balance(self) -> Dict[str, float]:
        """Fetches the current account balance for Coinbase."""
        url = f"{self.base_url}/v2/accounts"
        headers = self._get_signed_request_headers(url)
        response = requests.get(url, headers=headers)
        balance = {}
        for account in response.json()["data"]:
            if account["balance"]["amount"] != "0.00":
                balance[account["currency"]] = float(account["balance"]["amount"])
        return balance
