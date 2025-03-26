import json

import requests


class StandardBank:
    def __init__(self, client_id, client_secret, use_sandbox=True):
        self.base_url = "https://api.standardbank.co.za"
        self.client_id = client_id
        self.client_secret = client_secret
        self.use_sandbox = use_sandbox
        self.session = requests.Session()

        if use_sandbox:
            self.base_url = "https://api-sandbox.standardbank.co.za"

        self.session.auth = (client_id, client_secret)

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

    def get_access_token(self):
        endpoint = "/oauth/token"
        data = {
            "grant_type": "client_credentials",
            "scope": "accounts",
        }

        response = self.session.post(f"{self.base_url}/{endpoint}", data=data)

        if response.status_code != 200:
            raise Exception(
                f"Failed to get access token with status code {response.status_code}"
            )

        response_data = response.json()
        return response_data["access_token"]

    def get_accounts(self):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        endpoint = "/api/v1/accounts"
        return self.make_request(endpoint, headers=headers)

    def get_account_balance(self, account_id):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        endpoint = f"/api/v1/accounts/{account_id}/balances"
        return self.make_request(endpoint, headers=headers)

    def get_account_transactions(self, account_id, start_date=None, end_date=None):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        endpoint = f"/api/v1/accounts/{account_id}/transactions"

        params = {}

        if start_date is not None:
            params["fromDate"] = start_date.strftime("%Y-%m-%d")

        if end_date is not None:
            params["toDate"] = end_date.strftime("%Y-%m-%d")

        return self.make_request(endpoint, headers=headers, params=params)

    def transfer_funds(self, from_account_id, to_account_id, amount):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        endpoint = "/api/v1/payments/transfers"
        data = {
            "fromAccountId": from_account_id,
            "toAccountId": to_account_id,
            "amount": amount,
            "currency": "ZAR",
            "narrative": "Test transfer",
        }
        return self.make_request(
            endpoint, method="POST", headers=headers, data=json.dumps(data)
        )
