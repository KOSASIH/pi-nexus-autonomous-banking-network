import requests


class SmartHomeDevice:

    def __init__(self, device_id, device_token):
        self.device_id = device_id
        self.device_token = device_token

    def get_utility_bill(self):
        # Make a request to the smart home API to retrieve the utility bill
        response = requests.get(
            f"https://smart-home-api.com/bills/{self.device_id}",
            headers={"Authorization": f"Bearer {self.device_token}"},
        )
        return response.json()

    def make_payment(self, amount):
        # Make a request to the smart home API to make a payment
        response = requests.post(
            f"https://smart-home-api.com/payments/{self.device_id}",
            json={"amount": amount},
            headers={"Authorization": f"Bearer {self.device_token}"},
        )
        return response.json()
