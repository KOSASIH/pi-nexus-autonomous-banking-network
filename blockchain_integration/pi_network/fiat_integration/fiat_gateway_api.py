import requests

class FiatGatewayAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.fiat_gateway.com/v1"

    def get_fiat_exchange_rate(self, fiat_currency):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.base_url}/exchange_rate/{fiat_currency}", headers=headers)
        return response.json()

    def execute_swap(self, amount_fiat, fiat_currency):
        data = {
            "amount": amount_fiat,
            "currency": fiat_currency
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(f"{self.base_url}/swap", headers=headers, json=data)
        return response.json()
