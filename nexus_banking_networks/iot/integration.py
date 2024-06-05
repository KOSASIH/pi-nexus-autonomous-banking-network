import requests

class IoTIntegration:
    def __init__(self, iot_device):
        self.iot_device = iot_device

    def send_transaction(self, transaction_data):
        # Send transaction data to IoT device for real-time monitoring and automation
        response = requests.post(f'https://{iot_device}.com/api/transactions', json=transaction_data)
        return response.json()
