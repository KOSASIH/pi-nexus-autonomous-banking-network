import requests
import json
from config import config
from utils.logger import logger

class KYCService:
    def __init__(self):
        self.kyc_aml_api = config['kyc_aml_api']

    def verify_user(self, user_id, user_data):
        try:
            response = requests.post(self.kyc_aml_api, json=user_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifying user: {e}")
            return None
