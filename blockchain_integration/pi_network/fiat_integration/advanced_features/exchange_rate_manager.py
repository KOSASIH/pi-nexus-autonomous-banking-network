# exchange_rate_manager.py

import requests

class ExchangeRateManager:
    def __init__(self):
        self.exchange_rate_api = 'https://api.exchangerate-api.com/v4/latest/USD'

    def get_exchange_rate(self, currency):
        response = requests.get(self.exchange_rate_api)
        data = response.json()
        return data['rates'][currency]
