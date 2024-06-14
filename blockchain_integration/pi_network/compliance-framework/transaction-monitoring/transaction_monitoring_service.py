import requests
import json
from config import config
from utils.logger import logger

class TransactionMonitoringService:
    def __init__(self):
        self.transaction_monitoring_api = config['transaction_monitoring_api']

    def monitor_transaction(self, transaction_data):
        try:
            response = requests.post(self.transaction_monitoring_api, json=transaction_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error monitoring transaction: {e}")
            return None
