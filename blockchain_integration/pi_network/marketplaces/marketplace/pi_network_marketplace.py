from typing import Dict, Union

import requests


class PiNetworkMarketplace:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _get_request_headers(self) -> Dict[str, str]:
        """Returns the request headers for making requests to the Pi Network marketplace API."""
        return {
            "Content-Type": "application/json",
            "User-Agent": "Python/3.8 requests/2.25.1",
        }

    def fetch_products(self) -> Dict[str, Union[str, float]]:
        """Fetches the list of products from the Pi Network marketplace."""
        url = f"{self.base_url}/products"
        response = requests.get(url, headers=self._get_request_headers())
        products = {}
        for product in response.json():
            products[product["id"]] = {
                "name": product["name"],
                "price": float(product["price"]),
            }
        return products

    def place_order(
        self, order: Dict[str, Union[str, float]]
    ) -> Dict[str, Union[str, float]]:
        """Places an order on the Pi Network marketplace."""
        url = f"{self.base_url}/orders"
        response = requests.post(url, json=order, headers=self._get_request_headers())
        return response.json()

    def get_order_status(self, order_id: str) -> Dict[str, Union[str, float]]:
        """Fetches the status of an order on the Pi Network marketplace."""
        url = f"{self.base_url}/orders/{order_id}"
        response = requests.get(url, headers=self._get_request_headers())
        return response.json()

    def get_account_balance(self) -> Dict[str, Union[str, float]]:
        """Fetches the current account balance for the Pi Network marketplace."""
        url = f"{self.base_url}/balance"
        response = requests.get(url, headers=self._get_request_headers())
        balance = {}
        for asset in response.json():
            balance[asset["currency"]] = float(asset["balance"])
        return balance
