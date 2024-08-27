import requests
import json

# GDPR API keys
api_keys = {
    'data_protection': 'YOUR_API_KEY'
}

# Function to request data erasure
def request_data_erasure(user_id):
    url = 'https://api.data_protection.com/v1/erase'
    headers = {'x-apikey': api_keys['data_protection']}
    data = {'user_id': user_id}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Function to request data access
def request_data_access(user_id):
    url = 'https://api.data_protection.com/v1/access'
    headers = {'x-apikey': api_keys['data_protection']}
    data = {'user_id': user_id}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Example usage
user_id = '123456789'
erasure_result = request_data_erasure(user_id)
print(erasure_result)

access_result = request_data_access(user_id)
print(access_result)
