import requests
import json

class PiNetworkAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.minepi.com/v1"

    def get_user_balance(self, user_id):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.base_url}/users/{user_id}/balance", headers=headers)
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

    def get_fiat_exchange_rate(self, fiat_currency):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.base_url}/exchange_rates/{fiat_currency}", headers=headers)
        return response.json()
