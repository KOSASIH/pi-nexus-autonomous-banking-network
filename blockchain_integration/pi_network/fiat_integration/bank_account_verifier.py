import requests

class BankAccountVerifier:
    def __init__(self, bank_api_key, bank_api_secret):
        self.bank_api_key = bank_api_key
        self.bank_api_secret = bank_api_secret
        self.base_url = "https://api.bank.com/v1"

    def verify_bank_account(self, account_number, account_holder_name):
        headers = {
            "Authorization": f"Bearer {self.bank_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "account_number": account_number,
            "account_holder_name": account_holder_name
        }
        response = requests.post(f"{self.base_url}/verify_account", headers=headers, json=data)
        return response.json()
