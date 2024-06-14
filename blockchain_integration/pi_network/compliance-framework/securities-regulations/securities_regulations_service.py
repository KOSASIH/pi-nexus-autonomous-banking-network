import requests
import json
from config import config
from utils.logger import logger

class SecuritiesRegulationsService:
    def __init__(self):
        self.securities_regulations_api = config['securities_regulations_api']

    def verify_security(self, security_id):
        try:
            response = requests.get(self.securities_regulations_api + '/' + security_id)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifying security: {e}")
            return None
