import os
import requests
import json

# Set environment variables for cloud credentials
os.environ["CLOUD_API_KEY"] = "your_cloud_api_key"
os.environ["CLOUD_API_SECRET"] = "your_cloud_api_secret"

# Define function to send transaction data to cloud
def send_transaction_data_to_cloud(transaction_data):
    # Set cloud API endpoint
    cloud_api_endpoint = "https://your_cloud_api_endpoint.com/transaction_data"

    # Set cloud API headers
    cloud_api_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('CLOUD_API_KEY')}"
    }

    # Send transaction data to cloud
    cloud_api_response = requests.post(cloud_api_endpoint, headers=cloud_api_headers, data=json.dumps(transaction_data))

    # Check cloud API response status
    if cloud_api_response.status_code == 200:
        print("Transaction data sent to cloud successfully!")
    else:
        print(f"Error sending transaction data to cloud: {cloud_api_response.text}")

# Example usage
transaction_data = {
    "sender": "Alice",
    "receiver": "Bob",
    "amount": 100,
    "timestamp": "2023-05-23T10:00:00Z"
}

send_transaction_data_to_cloud(transaction_data)
