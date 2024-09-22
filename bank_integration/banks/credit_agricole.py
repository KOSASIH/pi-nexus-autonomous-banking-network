import requests
import json

class CreditAgricole:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.credit-agricole.com"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_account_info(self, account_number):
        url = f"{self.base_url}/accounts/{account_number}/info"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def make_transfer(self, account_number, recipient_account_number, amount):
        url = f"{self.base_url}/transfers"
        payload = {
            "account_number": account_number,
            "recipient_account_number": recipient_account_number,
            "amount": amount
        }
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json()

    def get_account_statement(self, account_number):
        url = f"{self.base_url}/accounts/{account_number}/statement"
        response = requests.get(url, headers=self.headers)
        return response.json()
