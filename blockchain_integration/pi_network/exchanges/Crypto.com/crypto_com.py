import requests
import json

class CryptoCom:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.crypto.com/v1'

    def get_balances(self):
        params = {
            'api_key': self.api_key
        }
        response = requests.get(self.base_url + '/account/balances', params=params)
        return response.json()

    def get_orders(self, symbol):
        params = {
            'api_key': self.api_key,
            'ymbol': symbol
        }
        response = requests.get(self.base_url + '/orders', params=params)
        return response.json()

    def place_order(self, symbol, type, quantity, price):
        params = {
            'api_key': self.api_key,
            'ymbol': symbol,
            'type': type,
            'amount': quantity,
            'price': price
        }
        response = requests.post(self.base_url + '/orders', params=params)
        return response.json()
