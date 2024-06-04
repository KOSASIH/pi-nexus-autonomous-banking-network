import requests
from .exceptions.pi_network_exceptions import PiNetworkApiException

class PiNetworkUtils:
    def __init__(self, api_endpoint, api_key, api_secret):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.api_secret = api_secret

    def get_blockchain_info(self):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        response = requests.get(f'{self.api_endpoint}/blockchain', headers=headers)
        if response.status_code != 200:
            raise PiNetworkApiException(response.status_code, response.text)
        return response.json()

    def broadcast_transaction(self, transaction):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        response = requests.post(f'{self.api_endpoint}/transaction', json=transaction, headers=headers)
        if response.status_code != 201:
            raise PiNetworkApiException(response.status_code, response.text)
        return response.json()

    def verify_transaction(self, transaction):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        response = requests.get(f'{self.api_endpoint}/transaction/{transaction["hash"]}', headers=headers)
        if response.status_code != 200:
            raise PiNetworkApiException(response.status_code, response.text)
        return response.json()
