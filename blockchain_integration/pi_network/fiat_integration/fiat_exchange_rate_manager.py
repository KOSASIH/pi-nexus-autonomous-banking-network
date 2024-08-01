import requests
import json

class FiatExchangeRateManager:
    def __init__(self, fiat_gateway_api_key, fiat_gateway_api_secret):
        self.fiat_gateway_api_key = fiat_gateway_api_key
        self.fiat_gateway_api_secret = fiat_gateway_api_secret
        self.base_url = "https://api.fiat_gateway.com/v1"

    def get_fiat_exchange_rates(self):
        headers = {
            "Authorization": f"Bearer {self.fiat_gateway_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.base_url}/exchange_rates", headers=headers)
        return response.json()

    def update_fiat_exchange_rates(self, exchange_rates):
        headers = {
            "Authorization": f"Bearer {self.fiat_gateway_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "exchange_rates": exchange_rates
        }
        response = requests.put(f"{self.base_url}/exchange_rates", headers=headers, json=data)
        return response.json()
