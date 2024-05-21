import json

import requests


class CIB:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.cib.com.eg"

    def get_account_balance(self, account_number):
        url = f"{self.base_url}/accounts/{account_number}/balance"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            raise Exception(f"Error getting account balance: {response.text}")

    def transfer_funds(self, from_account_number, to_account_number, amount):
        url = f"{self.base_url}/transfers"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "fromAccountNumber": from_account_number,
            "toAccountNumber": to_account_number,
            "amount": amount,
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            raise Exception(f"Error transferring funds: {response.text}")
