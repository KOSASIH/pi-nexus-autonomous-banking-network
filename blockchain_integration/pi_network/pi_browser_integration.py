import requests

class PiBrowserIntegration:
    def __init__(self, pi_browser_api_key):
        self.api_key = pi_browser_api_key

    def get_user_data(self, user_id):
        response = requests.get(f'https://pi-browser.io/api/v1/users/{user_id}', headers={'Authorization': f'Bearer {self.api_key}'})
        return response.json()

    def send_transaction(self, from_user_id, to_user_id, amount):
        response = requests.post(f'https://pi-browser.io/api/v1/transactions', headers={'Authorization': f'Bearer {self.api_key}'}, json={'from_user_id': from_user_id, 'to_user_id': to_user_id, 'amount': amount})
        return response.json()
