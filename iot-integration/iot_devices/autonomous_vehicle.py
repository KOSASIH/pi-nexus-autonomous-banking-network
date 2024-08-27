import requests

class AutonomousVehicleDevice:
    def __init__(self, device_id, device_token):
        self.device_id = device_id
        self.device_token = device_token

    def make_payment(self, amount):
        # Make a request to the autonomous vehicle API to make a payment
        response = requests.post(f'https://autonomous-vehicle-api.com/payments/{self.device_id}', json={'amount': amount}, headers={'Authorization': f'Bearer {self.device_token}'})
        return response.json()

    def get_vehicle_data(self):
        # Make a request to the autonomous vehicle API to retrieve vehicle data
        response = requests.get(f'https://autonomous-vehicle-api.com/vehicles/{self.device_id}', headers={'Authorization': f'Bearer {self.device_token}'})
        return response.json()
