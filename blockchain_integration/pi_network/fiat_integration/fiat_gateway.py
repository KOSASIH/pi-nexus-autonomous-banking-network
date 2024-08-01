import requests
import json

class FiatGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.fiat_gateway.com/v1"

    def get_fiat_rates(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.base_url}/rates", headers=headers)
        return response.json()

    def swap_pi_for_fiat(self, user_id, amount_pi, fiat_currency):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "user_id": user_id,
            "amount_pi": amount_pi,
            "fiat_currency": fiat_currency
        }
        response = requests.post(f"{self.base_url}/swap", headers=headers, json=data)
        return response.json()
