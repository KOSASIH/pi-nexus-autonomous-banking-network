import requests
import json
import hmac
import hashlib

class Bitfinex:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.bitfinex.com/v2'

    def get_balances(self):
        params = {
            'request': '/v2/auth/r/wallets',
            'nonce': int(time.time() * 1000)
        }
        headers = {
            'X-BFX-APIKEY': self.api_key
        }
        signature = hmac.new(self.api_secret.encode(), msg=json.dumps(params).encode(), digestmod=hashlib.sha384).hexdigest()
        headers['X-BFX-SIGNATURE'] = signature
        response = requests.get(self.base_url, params=params, headers=headers)
        return response.json()

    def get_orders(self, symbol):
        params = {
            'request': '/v2/auth/r/orders',
            'symbol': symbol,
            'limit': 100,
            'nonce': int(time.time() * 1000)
        }
        headers = {
            'X-BFX-APIKEY': self.api_key
        }
        signature = hmac.new(self.api_secret.encode(), msg=json.dumps(params).encode(), digestmod=hashlib.sha384).hexdigest()
        headers['X-BFX-SIGNATURE'] = signature
        response = requests.get(self.base_url, params=params, headers=headers)
        return response.json()

    def place_order(self, symbol, type, quantity, price):
        params = {
            'request': '/v2/auth/w/orders',
            'symbol': symbol,
            'amount': quantity,
            'price': price,
            'side': type,
            'type': 'EXCHANGE LIMIT',
            'nonce': int(time.time() * 1000)
        }
        headers = {
            'X-BFX-APIKEY': self.api_key
        }
        signature = hmac.new(self.api_secret.encode(), msg=json.dumps(params).encode(), digestmod=hashlib.sha384).hexdigest()
        headers['X-BFX-SIGNATURE'] = signature
        response = requests.post(self.base_url, params=params, headers=headers)
        return response.json()
