# kb_kookmin_bank.py
import requests
import json

class KBKookminBank:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.kbstar.com"

    def get_accounts(self):
        url = f"{self.base_url}/v1/accounts"
        headers = {"Authorization":f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        return json.loads(response.text)

    def get_transactions(self, account_id):
        url = f"{self.base_url}/v1/accounts/{account_id}/transactions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        return json.loads(response.text)

    def transfer_funds(self, from_account_id, to_account_id, amount):
        url = f"{self.base_url}/v1/transfers"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "fromAccountId": from_account_id,
            "toAccountId": to_account_id,
            "amount": amount
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return json.loads(response.text)
