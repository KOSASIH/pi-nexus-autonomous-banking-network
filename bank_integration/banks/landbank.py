import requests


class Landbank:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.landbank.com.ph"

    def get_account_balance(self, account_number):
        url = f"{self.base_url}/ibanking/accounts/{account_number}/balance"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        return response.json()

    def transfer_funds(self, from_account_number, to_account_number, amount):
        url = f"{self.base_url}/ibanking/transfers"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "fromAccountNumber": from_account_number,
            "toAccountNumber": to_account_number,
            "amount": amount,
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
