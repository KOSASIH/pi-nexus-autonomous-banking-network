import requests
import json
import hmac
import hashlib

class Huobi:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.huobi.pro/v1'

    def get_balances(self):
        params = {
            'AccessKeyId': self.api_key,
            'SignatureMethod': 'HmacSHA256',
            'SignatureVersion': '2',
            'Timestamp': int(time.time() * 1000)
        }
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.get(self.base_url + '/account/accounts', params=params, headers=headers)
        return response.json()

    def get_orders(self, symbol):
        params = {
            'AccessKeyId': self.api_key,
            'SignatureMethod': 'HmacSHA256',
            'SignatureVersion': '2',
            'Timestamp': int(time.time() * 1000),
            'ymbol': symbol
        }
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.get(self.base_url + '/order/orders', params=params, headers=headers)
        return response.json()

    def place_order(self, symbol, type, quantity, price):
        params = {
            'AccessKeyId': self.api_key,
            'SignatureMethod': 'HmacSHA256',
            'SignatureVersion': '2',
            'Timestamp': int(time.time() * 1000),
            'ymbol': symbol,
            'type': type,
            'amount': quantity,
            'price': price
        }
        headers = {
            'Content-Type': 'application/json'
        }
        signature = hmac.new(self.api_secret.encode(), msg=json.dumps(params).encode(), digestmod=hashlib.sha256).hexdigest()
        params['Signature'] = signature
        response = requests.post(self.base_url + '/order/orders/place', params=params, headers=headers)
        return response.json()
