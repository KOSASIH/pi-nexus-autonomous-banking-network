import requests
import json
from config import config
from utils.logger import logger

class PCIDSSService:
    def __init__(self):
        self.pci_dss_api = config['pci_dss_api']

    def verify_pci_dss(self, pci_dss_id):
        try:
            response = requests.get(self.pci_dss_api + '/' + pci_dss_id)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifyingPCI DSS: {e}")
            return None
