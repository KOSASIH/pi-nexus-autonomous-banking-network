import requests
import json

class FNBBank:
    def __init__(self, username, password):
        self.base_url = "https://api.fnb.co.za/v1"
        self.session = requests.Session()
        self.session.auth = (username, password)

    def make_request(self, endpoint, method="GET", data=None, headers=None):
        url = f"{self.base_url}/{endpoint}"

        if headers is None:
            headers = {
                "Content-Type": "application/json",
            }

        response = self.session.request(method, url, data=data, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")

        return response.json()

    def get_accounts(self):
        endpoint = "/accounts"
        return self.make_request(endpoint)

    def get_transactions(self, account_number):
        endpoint = f"/transactions/{account_number}"
        return self.make_request(endpoint)

    def get_balance(self, account_number):
        endpoint = f"/balance/{account_number}"
        return self.make_request(endpoint)

    def transfer_funds(self, from_account_number, to_account_number, amount):
        endpoint = "/transfers"
        data = {
            "fromAccountNumber": from_account_number,
            "toAccountNumber": to_account_number,
            "amount": amount,
            "currency": "ZAR",
            "narrative": "Test transfer",
        }
        return self.make_request(endpoint, method="POST", data=json.dumps(data))
