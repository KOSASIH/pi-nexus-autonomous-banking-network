import requests


class WearableDevice:

    def __init__(self, device_id, device_token):
        self.device_id = device_id
        self.device_token = device_token

    def authenticate_transaction(self, transaction_id):
        # Make a request to the wearable API to authenticate the transaction
        response = requests.get(
            f"https://wearable-api.com/transactions/{transaction_id}",
            headers={"Authorization": f"Bearer {self.device_token}"},
        )
        return response.json()

    def get_user_data(self):
        # Make a request to the wearable API to retrieve user data
        response = requests.get(
            f"https://wearable-api.com/users/{self.device_id}",
            headers={"Authorization": f"Bearer {self.device_token}"},
        )
        return response.json()
