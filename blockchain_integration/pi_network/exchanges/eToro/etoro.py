import requests
import json

class eToro:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.etoro.com/v1'

    def get_balances(self):
        params = {
            'access_token': self.api_key
        }
        response = requests.get(self.base_url + '/accounts/balances', params=params)
        return response.json()

    def get_orders(self, symbol):
        params = {
            'access_token': self.api_key,
            'ymbol': symbol
        }
        response = requests.get(self.base_url + '/orders', params=params)
        return response.json()

    defplace_order(self, symbol, type, quantity, price):
        params = {
            'access_token': self.api_key,
            'ymbol': symbol,
            'type': type,
            'amount': quantity,
            'price': price
        }
        response = requests.post(self.base_url + '/orders', params=params)
        return response.json()
