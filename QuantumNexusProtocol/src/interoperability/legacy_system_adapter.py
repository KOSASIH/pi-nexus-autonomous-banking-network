import requests

class LegacySystemAdapter:
    def __init__(self, legacy_api_url):
        self.legacy_api_url = legacy_api_url

    def fetch_data(self, endpoint):
        response = requests.get(f"{self.legacy_api_url}/{endpoint}")
        return response.json()

    def send_data(self, endpoint, data):
        response = requests.post(f"{self.legacy_api_url}/{endpoint}", json=data)
        return response.status_code, response.json()

# Example usage
if __name__ == "__main__":
    adapter = LegacySystemAdapter('https://legacy-system-url/api')
    data = adapter.fetch_data('data-endpoint')
    print(f"Fetched Data: {data}")
    status, response = adapter.send_data('data-endpoint', {'key': 'value'})
    print(f"Response Status: {status}, Response: {response}")
