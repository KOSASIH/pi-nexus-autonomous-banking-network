import requests
import json

class FiatExchangeRateUpdater:
    def __init__(self, fiat_gateway_api_key, fiat_gateway_api_secret):
        self.fiat_gateway_api_key = fiat_gateway_api_key
        self.fiat_gateway_api_secret = fiat_gateway_api_secret
        self.base_url = "https://api.fiat_gateway.com/v1"

    def update_fiat_exchange_rate(self, fiat_currency):
        headers = {
            "Authorization": f"Bearer {self.fiat_gateway_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.base_url}/exchange_rates/{fiat_currency}", headers=headers)
        return response.json()
