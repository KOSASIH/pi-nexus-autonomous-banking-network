import requests
import json
from config import config
from utils.logger import logger

class TravelRuleService:
    def __init__(self):
        self.travel_rule_api = config['travel_rule_api']

    def verify_transaction(self, transaction_data):
        try:
            response = requests.post(self.travel_rule_api, json=transaction_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifying transaction: {e}")
            return None
