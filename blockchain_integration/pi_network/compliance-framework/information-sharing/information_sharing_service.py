import requests
import json
from config import config
from utils.logger import logger

class InformationSharingService:
    def __init__(self):
        self.information_sharing_api = config['information_sharing_api']

    def share_information(self, information_id):
        try:
            response = requests.post(self.information_sharing_api, json={'information_id': information_id})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sharing information: {e}")
            return None
