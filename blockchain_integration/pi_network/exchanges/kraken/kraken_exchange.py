import requests
from typing import Dict, Union

class KrakenExchange:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.kraken.com"

    def _get_signed_request_headers(self, endpoint: str) -> Dict[str, str]:
        """Returns the signed request headers for a given endpoint."""
        nonce = str(int(requests.get("https://api.kraken.com/0/time").json()["epoch"]))
        message = f"{nonce}{endpoint}"
        secret_key = self.api_secret.encode("utf-8")
        signature = requests.hmac.new(secret_key, message.encode("utf-8"), hashlib.sha256).hexdigest()
        return {
            "API-Key": self.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Python/3.8 requests/2.25.1",
        }

    def fetch_exchange_rates(self) -> Dict[str, float]:
        """Fetches the current exchange rates for Kraken."""
        url = f"{self.base_url}/0/public/Ticker"
        response = requests.get(url)
        exchange_rates = {}
        for rate in response.json()["result"]:
            if rate["wsname"] == "XBTUSD":
                exchange_rates["BTC"] = float(rate["c"][0])
                exchange_rates["USD"] = float(rate["c"][1])
        return exchange_rates

    def place_order(self, order: Dict[str, Union[str, float]]) -> Dict[str, Union[str, float]]:
        """Places an order on Kraken."""
        url = f"{self.base_url}/0/private/Order/Place"
        headers = self._get_signed_request_headers(url)
        response = requests.post(url, data=order, headers=headers)
        return response.json()

    def get_account_balance(self) -> Dict[str, float]:
        """Fetches the current account balance for Kraken."""
        url = f"{self.base_url}/0/private/Balance"
        headers = self._get_signed_request_headers(url)
        response = requests.post(url, headers=headers)
        balance = {}
        for asset in response.json()["result"]:
            if asset["acctype"] == "trading":
                balance[asset["currency"]] = float(asset["balance"])
        return balance
