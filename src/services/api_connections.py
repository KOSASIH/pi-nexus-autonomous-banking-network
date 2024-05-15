# src/services/api_connections.py
import requests


class PiNexusAPI:
    def __init__(self):
        self.base_url = "https://api.pi-nexus.com"

    def get_data(self):
        response = requests.get(self.base_url + "/data")
        return response.json()
