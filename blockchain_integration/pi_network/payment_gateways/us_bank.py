# payment_gateways/us_bank.py
import requests


class USBankPaymentGateway:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://api.usbank.com/v1"

    def make_payment(self, amount, recipient_account_number):
        headers = {
            "Authorization": f"Bearer {self.client_id}",
            "Content-Type": "application/json",
        }
        data = {"amount": amount, "recipient_account_number": recipient_account_number}
        response = requests.post(
            f"{self.base_url}/payments", headers=headers, json=data
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Payment failed")
