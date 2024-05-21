import json

import requests


class CSBBank:
    def __init__(self, base_url, client_id, client_secret, access_token=None):
        self.base_url = base_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.session = requests.Session()

    def get_access_token(self):
        if self.access_token is None:
            self.access_token = self._get_access_token_from_api()
        return self.access_token

    def _get_access_token_from_api(self):
        url = f"{self.base_url}/oauth/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        response = self.session.post(url, headers=headers, data=data)
        response_data = response.json()
        access_token = response_data["access_token"]
        return access_token

    def get_accounts(self):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.base_url}/api/v1/accounts"
        response = self.session.get(url, headers=headers)
        response_data = response.json()
        return response_data

    def get_account_balance(self, account_id):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.base_url}/api/v1/accounts/{account_id}/balance"
        response = self.session.get(url, headers=headers)
        response_data = response.json()
        return response_data

    def get_account_transactions(self, account_id, start_date, end_date):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.base_url}/api/v1/accounts/{account_id}/transactions"
        params = {"startDate": start_date, "endDate": end_date}
        response = self.session.get(url, headers=headers, params=params)
        response_data = response.json()
        return response_data

    def transfer_funds(self, from_account_id, to_account_id, amount):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.base_url}/api/v1/transfers"
        data = {
            "fromAccountId": from_account_id,
            "toAccountId": to_account_id,
            "amount": amount,
        }
        response = self.session.post(url, headers=headers, data=json.dumps(data))
        response_data = response.json()
        return response_data
