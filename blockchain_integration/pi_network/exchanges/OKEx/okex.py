import requests
import json
import hmac
import hashlib

class OKEx:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://www.okex.com/api/v5'

    def get_balances(self):
        params = {
            'api_key': self.api_key,
            'timestamp': int(time.time() * 1000)
        }
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self._sign(params),
            'OK-ACCESS-TIMESTAMP': int(time.time() * 1000),
            'OK-ACCESS-PASSPHRASE': self.api_secret
        }
        response = requests.get(self.base_url + '/account/balance', params=params, headers=headers)
        return response.json()

    def get_orders(self, symbol):
        params = {
            'api_key': self.api_key,
            'symbol': symbol,
            'timestamp': int(time.time() * 1000)
        }
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self._sign(params),
            'OK-ACCESS-TIMESTAMP': int(time.time() * 1000),
            'OK-ACCESS-PASSPHRASE': self.api_secret
        }
        response = requests.get(self.base_url + '/order/list', params=params, headers=headers)
        return response.json()

    def place_order(self, symbol, type, quantity, price):
        params = {
            'api_key': self.api_key,
            'symbol': symbol,
            'type': type,
            'amount': quantity,
            'price': price,
            'timestamp': int(time.time() * 1000)
        }
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self._sign(params),
            'OK-ACCESS-TIMESTAMP': int(time.time() * 1000),
            'OK-ACCESS-PASSPHRASE': self.api_secret
        }
        response = requests.post(self.base_url + '/order/place', params=params, headers=headers)
        return response.json()

    def _sign(self, params):
        # OKEx's API signature calculation
        pass
