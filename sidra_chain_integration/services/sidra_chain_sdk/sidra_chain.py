import requests

class SidraChain:
    def __init__(self):
        self.base_url = "https://sidra-chain.com/api"

    def create_transaction(self, transaction: dict):
        response = requests.post(f"{self.base_url}/transactions", json=transaction)
        response.raise_for_status()
        return response.json()
