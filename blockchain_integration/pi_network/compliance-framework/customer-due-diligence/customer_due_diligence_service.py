import requests
import json
from config import config
from utils.logger import logger

class CustomerDueDiligenceService:
    def __init__(self):
        self.customer_due_diligence_api = config['customer_due_diligence_api']

    def perform_due_diligence(self, user_id, user_data):
        try:
            response = requests.post(self.customer_due_diligence_api, json={'user_id': user_id, 'user_data': user_data})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error performing due diligence: {e}")
            return None
