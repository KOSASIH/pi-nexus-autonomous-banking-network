# microservices/banking_service.py
import requests


class BankingService:
    def __init__(self, config):
        self.config = config

    def get_banking_data(self):
        # Call external API or database to retrieve banking data
        response = requests.get(self.config.banking_api_url)
        return response.json()
