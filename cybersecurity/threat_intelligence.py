import requests
import json
from datetime import datetime

# Threat Intelligence API keys
api_keys = {
    'virustotal': 'YOUR_API_KEY',
    'alienvault': 'YOUR_API_KEY',
    'threatcrowd': 'YOUR_API_KEY'
}

# Function to fetch threat intelligence data
def fetch_threat_intel(ip_address):
    intel_data = {}
    for api, key in api_keys.items():
        if api == 'virustotal':
            url = f'https://www.virustotal.com/api/v3/ip_addresses/{ip_address}'
            headers = {'x-apikey': key}
            response = requests.get(url, headers=headers)
            intel_data[api] = response.json()
        elif api == 'alienvault':
            url = f'https://otx.alienvault.com/api/v1/indicators/ip/{ip_address}'
            headers = {'x-apikey': key}
            response = requests.get(url, headers=headers)
            intel_data[api] = response.json()
        elif api == 'threatcrowd':
            url = f'https://www.threatcrowd.org/api/v2/ip/{ip_address}'
            response = requests.get(url)
            intel_data[api] = response.json()
    return intel_data

# Example usage
ip_address = '8.8.8.8'
intel_data = fetch_threat_intel(ip_address)
print(json.dumps(intel_data, indent=4))
