import requests

class IoTIntegration:
    def __init__(self):
        self.api_url = "https://iot-device-api.com/transactions"

    def send_transaction(self, transaction_data):
        response = requests.post(self.api_url, json=transaction_data)
        if response.status_code == 200:
            return True
        else:
            return False

# Example usage:
iot_integration = IoTIntegration()
transaction_data = {"amount": 100, "recipient": "User 1"}
if iot_integration.send_transaction(transaction_data):
    print("Transaction sent successfully!")
else:
    print("Transaction failed!")
