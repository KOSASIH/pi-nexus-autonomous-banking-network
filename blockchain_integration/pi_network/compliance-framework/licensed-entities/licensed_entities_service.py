import requests
import json
from config import config
from utils.logger import logger

class LicensedEntitiesService:
    def __init__(self):
        self.licensed_entities_api = config['licensed_entities_api']

    def verify_licensed_entity(self, entity_id):
        try:
            response = requests.get(self.licensed_entities_api + '/' + entity_id)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifying licensed entity: {e}")
            return None
