import requests
import json
from config import config
from utils.logger import logger

class FraudDetectionService:
    def __init__(self):
        self.fraud_detection_api = config['fraud_detection_api']

    def detect_fraud(self, transaction_data):
        try:
            response = requests.post(self.fraud_detection_api, json=transaction_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error detecting fraud: {e}")
            return None
