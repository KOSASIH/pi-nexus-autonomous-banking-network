import requests
import json

class CapitecBank:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.capitecbank.co.za/v1"

    def make_request(self, endpoint, method="GET", data=None, headers=None):
        url = f"{self.base_url}/{endpoint}"

        if headers is None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.get_access_token()}",
            }

        response = requests.request(method, url, data=data, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")

        return response.json()

    def get_access_token(self):
        endpoint = "/oauth/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "scope": "openbanking",
        }

        response = requests.post(f"{self.base_url}/{endpoint}", data=data)

        if response.status_code != 200:
            raise Exception(f"Failed to get access token with status code {response.status_code}")

        response_data = response.json()
        return response_data["access_token"]

    def get_account_balance(self, account_id):
        endpoint = f"/accounts/{account_id}/balances"
        return self.make_request(endpoint)

    def transfer_funds(self, from_account_id, to_account_id, amount):
        endpoint = "/payments/transfers"
        data = {
            "fromAccountId": from_account_id,
            "toAccountId": to_account_id,
            "amount": amount,
            "currency": "ZAR",
            "narrative": "Test transfer",
        }
        return self.make_request(endpoint, method="POST", data=json.dumps(data))
