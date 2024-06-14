import requests
import json
from config import config
from utils.logger import logger

class AuditingCertificationService:
    def __init__(self):
        self.auditing_certification_api = config['auditing_certification_api']

    def verify_audit(self, audit_id):
        try:
            response = requests.get(self.auditing_certification_api + '/' + audit_id)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifying audit: {e}")
            return None
