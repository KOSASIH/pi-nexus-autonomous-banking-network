import requests
import json
from config import config
from utils.logger import logger

class ContinuousMonitoringService:
    def __init__(self):
        self.continuous_monitoring_api = config['continuous_monitoring_api']

    def monitor_activity(self, user_id, transaction_data):
        try:
            response = requests.post(self.continuous_monitoring_api, json={'user_id': user_id, 'transaction_data': transaction_data})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error monitoring activity: {e}")
            return None
