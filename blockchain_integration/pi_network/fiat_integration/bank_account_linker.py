import requests
import json

class BankAccountLinker:
    def __init__(self, bank_api_key, bank_api_secret):
        self.bank_api_key = bank_api_key
        self.bank_api_secret = bank_api_secret
        self.base_url = "https://api.bank.com/v1"

    def link_bank_account(self, user_id, account_number, account_holder_name):
        headers = {
            "Authorization": f"Bearer {self.bank_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "user_id": user_id,
            "account_number": account_number,
            "account_holder_name": account_holder_name
        }
        response = requests.post(f"{self.base_url}/link_account", headers=headers, json=data)
        return response.json()
