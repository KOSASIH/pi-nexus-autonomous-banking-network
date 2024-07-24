import requests

class InternetOfThingsDevice:
    def __init__(self, device_id, api_key):
        self.device_id = device_id
        self.api_key = api_key

    def send_data(self, data):
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.post(f'https://iot-api.example.com/devices/{self.device_id}/data', headers=headers, json=data)
        return response.json()

    def receive_data(self):
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get(f'https://iot-api.example.com/devices/{self.device_id}/data', headers=headers)
        return response.json()
