import requests
import json
from config import config
from utils.logger import logger

class AMLWatchlistService:
    def __init__(self):
        self.aml_watchlist_api = config['aml_watchlist_api']

    def check_watchlist(self, user_id, transaction_data):
        try:
            response = requests.post(self.aml_watchlist_api, json={'user_id': user_id, 'transaction_data': transaction_data})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking watchlist: {e}")
            return None
