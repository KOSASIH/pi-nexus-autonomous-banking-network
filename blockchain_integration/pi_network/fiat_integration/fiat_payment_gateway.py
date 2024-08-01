import requests
import json

class FiatPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.fiat_gateway.com/v1"

    def process_fiat_payment(self, user_id, amount_fiat, fiat_currency):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "user_id": user_id,
            "amount_fiat": amount_fiat,
            "fiat_currency": fiat_currency
        }
        response = requests.post(f"{self.base_url}/process_payment", headers=headers, json=data)
        return response.json()
