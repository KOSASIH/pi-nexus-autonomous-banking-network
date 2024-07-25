import requests
import json

class SidraChainConnector:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def authenticate(self):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        response = requests.post(f'{self.base_url}/auth', headers=headers, json={'api_secret': self.api_secret})
        if response.status_code == 200:
            self.access_token = response.json()['access_token']
            return True
        return False

    def get_data_catalog(self):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        response = requests.get(f'{self.base_url}/data_catalog', headers=headers)
        if response.status_code == 200:
            return response.json()
        return None

    def get_data_quality(self, dataset_id):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        response = requests.get(f'{self.base_url}/data_quality/{dataset_id}', headers=headers)
        if response.status_code == 200:
            return response.json()
        return None

    def get_data_lineage(self, dataset_id):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        response = requests.get(f'{self.base_url}/data_lineage/{dataset_id}', headers=headers)
        if response.status_code == 200:
            return response.json()
        return None

    def get_data_security(self, dataset_id):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        response = requests.get(f'{self.base_url}/data_security/{dataset_id}', headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
