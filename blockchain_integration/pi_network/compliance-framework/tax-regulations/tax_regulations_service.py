import requests
import json
from config import config
from utils.logger import logger

class TaxRegulationsService:
    def __init__(self):
        self.tax_regulations_api = config['tax_regulations_api']

    def verify_tax(self, tax_id):
        try:
            response = requests.get(self.tax_regulations_api + '/' + tax_id)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifying tax: {e}")
            return None
