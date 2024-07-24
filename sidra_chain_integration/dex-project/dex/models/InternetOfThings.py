import requests

class InternetOfThings:
    def __init__(self, api_key):
        self.api_key = api_key

    def send_data(self, data):
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.post('https://iot-api.example.com/data', headers=headers, json=data)
        return response.json()

    def receive_data(self):
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get('https://iot-api.example.com/data', headers=headers)
        return response.json()
