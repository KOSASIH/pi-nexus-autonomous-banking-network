import requests
import json
from config import config
from utils.logger import logger

class SanctionsListService:
    def __init__(self):
        self.sanctions_list_api = config['sanctions_list_api']

    def check_sanctions(self, user_id, transaction_data):
        try:
            response = requests.post(self.sanctions_list_api, json={'user_id': user_id, 'transaction_data': transaction_data})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking sanctions: {e}")
            return None
