# sidra_chain_api.py
import requests

class SidraChainAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def authenticate(self):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        response = requests.post(f'https://api.sidra.com/authenticate', headers=headers, json={'api_secret': self.api_secret})
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            return None

    def get_chain_data(self, access_token):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        response = requests.get(f'https://api.sidra.com/chain/data', headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None
