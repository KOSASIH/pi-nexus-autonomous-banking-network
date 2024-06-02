# payment_gateways/alipay.py
import requests


class AlipayPaymentGateway:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.alipay.com/v1"

    def make_payment(self, amount, recipient_account_number):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
