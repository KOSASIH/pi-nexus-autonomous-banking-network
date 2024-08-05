import requests
import json

class BankAPIConnector:
    def __init__(self, bank_api_key, bank_api_secret):
        self.bank_api_key = bank_api_key
        self.bank_api_secret = bank_api_secret
        self.base_url = "https://api.bank.com/v1"

    def get_bank_account_balance(self, account_number):
        headers = {
            "Authorization": f"Bearer {self.bank_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.base_url}/accounts/{account_number}/balance", headers=headers)
        return response.json()

    def transfer_fiat_to_bank_account(self, account_number, amount_fiat):
        headers = {
            "Authorization": f"Bearer {self.bank_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "account_number": account_number,
            "amount_fiat": amount_fiat
        }
        response = requests.post(f"{self.base_url}/transfer", headers=headers, json=data)
        return response.json()
