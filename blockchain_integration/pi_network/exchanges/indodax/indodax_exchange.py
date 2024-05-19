from typing import Dict, Union

import requests


class IndodaxExchange:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.indodax.com"

    def _get_signed_request_headers(self, endpoint: str) -> Dict[str, str]:
        """Returns the signed request headers for a given endpoint."""
        nonce = str(
            int(requests.get("https://api.indodax.com/v1/time").json()["server_time"])
        )
        message = f"{nonce}{endpoint}"
        secret_key = self.api_secret.encode("utf-8")
        signature = requests.hmac.new(
            secret_key, message.encode("utf-8"), hashlib.sha512
        ).hexdigest()
        return {
            "Key": self.api_key,
            "Sign": signature,
            "Nonce": nonce,
        }

    def fetch_exchange_rates(self) -> Dict[str, float]:
        """Fetches the current exchange rates for Indodax."""
        url = f"{self.base_url}/ticker"
        response = requests.get(url)
        exchange_rates = {}
        for rate in response.json():
            if rate["market"] == "BTCIDR":
                exchange_rates["BTC"] = float(rate["last_price"])
                exchange_rates["IDR"] = float(rate["last_price"]) * -1
        return exchange_rates

    def place_order(
        self, order: Dict[str, Union[str, float]]
    ) -> Dict[str, Union[str, float]]:
        """Places an order on Indodax."""
        url = f"{self.base_url}/order/book"
        headers = self._get_signed_request_headers(url)
        response = requests.post(url, json=order, headers=headers)
        return response.json()

    def get_account_balance(self) -> Dict[str, float]:
        """Fetches the current account balance for Indodax."""
        url = f"{self.base_url}/balance"
        headers = self._get_signed_request_headers(url)
        response = requests.get(url, headers=headers)
        balance = {}
        for asset in response.json():
            if asset["currency"] == "BTC":
                balance[asset["currency"]] = float(asset["available"])
            elif asset["currency"] == "IDR":
                balance["IDR"] = float(asset["available"])
        return balance
