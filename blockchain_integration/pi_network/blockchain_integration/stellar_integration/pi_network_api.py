import requests

class PiNetworkAPI:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def get_user_balance(self, user_id: str) -> int:
        """Return the Pi balance for the specified user"""
        response = requests.get(f"{self.api_url}/users/{user_id}/balance", headers={"Authorization": f"Bearer {self.api_key}"})
        return response.json()["balance"]

    def send_pi_payment(self, source_user_id: str, destination_user_id: str, amount: int) -> str:
        """Send a Pi payment from the source user to the destination user"""
        response = requests.post(f"{self.api_url}/payments", headers={"Authorization": f"Bearer {self.api_key}"}, json={
            "source_user_id": source_user_id,
            "destination_user_id": destination_user_id,
            "amount": amount
        })
        return response.json()["transaction_id"]
