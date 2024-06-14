import requests
import json
from config import config
from utils.logger import logger

class DataProtectionService:
    def __init__(self):
        self.data_protection_api = config['data_protection_api']

    def protect_user_data(self, user_id, user_data):
        try:
            response = requests.post(self.data_protection_api, json={'user_id': user_id, 'user_data': user_data})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error protecting user data: {e}")
            return None
