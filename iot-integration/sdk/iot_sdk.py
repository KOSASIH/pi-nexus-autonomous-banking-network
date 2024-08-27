import requests

class IotSdk:
    def __init__(self, api_url):
        self.api_url = api_url

    def register_device(self, device_type, device_id, device_token):
        # Register a new IoT device
        response = requests.post(f'{self.api_url}/iot/devices/{device_type}', json={'device_id': device_id, 'device_token': device_token})
        return response.json()

    def make_transaction(self, device_id, transaction_id, amount):
        # Make a transaction using an IoT device
        response = requests.post(f'{self.api_url}/iot/transactions', json={'device_id': device_id, 'transaction_id': transaction_id, 'amount': amount})
        return response.json()
