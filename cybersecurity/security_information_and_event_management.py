import requests
import json

# SIEM API keys
api_keys = {
    'log_collection': 'YOUR_API_KEY',
    'threat_detection': 'YOUR_API_KEY'
}

# Function to collect logs
def collect_logs(log_data):
    url = 'https://api.log_collection.com/v1/collect'
    headers = {'x-apikey': api_keys['log_collection']}
    response = requests.post(url, headers=headers, json=log_data)
    return response.json()

# Function to detect threats
def detect_threats(log_data):
    url = 'https://api.threat_detection.com/v1/detect'
    headers = {'x-apikey': api_keys['threat_detection']}
    response = requests.post(url, headers=headers, json=log_data)
    return response.json()

# Example usage
log_data = {'system_ip': '192.168.1.100', 'log_data': '...'}
collected_logs = collect_logs(log_data)
print(collected_logs)

detected_threats = detect_threats(log_data)
print(detected_threats)
