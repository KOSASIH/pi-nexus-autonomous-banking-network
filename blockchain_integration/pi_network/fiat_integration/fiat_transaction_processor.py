import requests
import json

class FiatTransactionProcessor:
    def __init__(self, fiat_gateway_api_key, fiat_gateway_api_secret):
        self.fiat_gateway_api_key = fiat_gateway_api_key
        self.fiat_gateway_api_secret = fiat_gateway_api_secret
        self.base_url = "https://api.fiat_gateway.com/v1"

    def process_fiat_transaction(self, user_id, amount_fiat, fiat_currency):
        headers = {
            "Authorization": f"Bearer {self.fiat_gateway_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "user_id": user_id,
            "amount_fiat": amount_fiat,
            "fiat_currency": fiat_currency
        }
        response = requests.post(f"{self.base_url}/transactions", headers=headers, json=data)
        return response.json()
