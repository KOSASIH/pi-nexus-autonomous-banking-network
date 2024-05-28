# pi_network_api.py

import json

import requests


class PiNetworkAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.minepi.com/v2"

    def get_balance(self, address):
        url = f"{self.base_url}/balance/{address}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()["balance"]
        else:
            raise Exception(f"Error fetching balance: {response.status_code}")

    def get_transaction(self, tx_id):
        url = f"{self.base_url}/transaction/{tx_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error fetching transaction: {response.status_code}")

    def submit_transaction(self, from_address, to_address, amount):
        url = f"{self.base_url}/transaction"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "from_address": from_address,
            "to_address": to_address,
            "amount": amount,
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["tx_id"]
        else:
            raise Exception(f"Error submitting transaction: {response.status_code}")


# usage
api = PiNetworkAPI("your_api_key_here")
balance = api.get_balance("your_address_here")
print(f"Balance: {balance}")

tx_id = api.submit_transaction("from_address_here", "to_address_here", 100)
print(f"Transaction ID: {tx_id}")

tx_info = api.get_transaction(tx_id)
print(f"Transaction Info: {tx_info}")
