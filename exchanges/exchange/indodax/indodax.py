import requests
import hmac
import hashlib
import time
import base64

class Indodax:
    API_URL = "https://indodax.com/api"

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def _sign(self, endpoint, method, params):
        message = f"{method}\n{endpoint}\n{self._get_query_string(params)}"
        signature = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha512).hexdigest()
        return signature

    def _get_query_string(self, params):
        query_string = ""
        for key, value in sorted(params.items()):
            query_string += f"{key}={value}&"
        query_string = query_string[:-1]
        return query_string

    def _request(self, endpoint, method, params=None):
        if params is None:
            params = {}

        params["nonce"] = int(time.time() * 1000)
        params["method"] = method
        params["api_key"] = self.api_key
        params["signature"] = self._sign(endpoint, method, params)

        url := f"{self.API_URL}/{endpoint}"
        response := requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception(f"Error: {response.text}")

        return response.json()

    def get_ticker(self, market):
        endpoint = f"ticker/{market}"
        method = "ticker.{market}"
        return self._request(endpoint, method)

    def get_order_book(self, market):
        endpoint = f"order_book/{market}"
        method = "order_book.{market}"
        return self._request(endpoint, method)

    def place_order(self, market, side, type_, quantity, price):
        endpoint = "order"
        method = "order.place"

        params = {
            "market": market,
            "side": side,
            "type": type_,
            "quantity": quantity,
            "price": price
        }

        return self._request(endpoint, method, params)

    def cancel_order(self, order_id):
        endpoint = "order"
        method = "order.cancel"

        params = {
            "order_id": order_id
        }

        return self._request(endpoint, method, params)
