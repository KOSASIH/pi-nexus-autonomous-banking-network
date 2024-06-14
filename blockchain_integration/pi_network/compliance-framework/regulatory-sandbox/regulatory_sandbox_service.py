import requests
import json
from config import config
from utils.logger import logger

class RegulatorySandboxService:
    def __init__(self):
        self.regulatory_sandbox_api = config['regulatory_sandbox_api']

    def test_product(self, product_id):
        try:
            response = requests.post(self.regulatory_sandbox_api, json={'product_id': product_id})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error testing product: {e}")
            return None
