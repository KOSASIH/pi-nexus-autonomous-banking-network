import requests


class BankIntegration:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def make_request(self, endpoint, params=None):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(
            f"{self.base_url}/{endpoint}", headers=headers, params=params
        )
        return response.json()

    def get_account_balance(self, account_number):
        response = self.make_request(
            "accounts/balance", {"account_number": account_number}
        )
        return response["balance"]

    def transfer_funds(self, from_account_number, to_account_number, amount):
        response = self.make_request(
            "transfers",
            {
                "from_account_number": from_account_number,
                "to_account_number": to_account_number,
                "amount": amount,
            },
        )
        return response["status"]
