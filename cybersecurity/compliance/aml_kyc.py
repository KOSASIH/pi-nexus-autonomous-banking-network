import requests
import json

# AML/KYC API keys
api_keys = {
    'sanctionscreening': 'YOUR_API_KEY',
    'identityverification': 'YOUR_API_KEY'
}

# Function to check sanctions screening
def check_sanctions(name, country):
    url = 'https://api.sanctionscreening.com/v1/check'
    headers = {'x-apikey': api_keys['sanctionscreening']}
    data = {'name': name, 'country': country}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Function to verify identity
def verify_identity(id_number, id_type, country):
    url = 'https://api.identityverification.com/v1/verify'
    headers = {'x-apikey': api_keys['identityverification']}
    data = {'id_number': id_number, 'id_type': id_type, 'country': country}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Example usage
name = 'John Doe'
country = 'US'
sanctions_result = check_sanctions(name, country)
print(sanctions_result)

id_number = '123456789'
id_type = 'passport'
country = 'US'
identity_result = verify_identity(id_number, id_type, country)
print(identity_result)
