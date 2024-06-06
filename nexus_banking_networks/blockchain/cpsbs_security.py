# cpsbs_security.py
import numpy as np
from cyber_physical_systems import CyberPhysicalSystems

class CPSBS:
    def __init__(self):
        self.cps = CyberPhysicalSystems()

    def secure_blockchain(self, blockchain_data):
        secured_data = self.cps.secure(blockchain_data)
        return secured_data

    def detect_anomalies(self, blockchain_data):
        anomalies = self.cps.detect_anomalies(blockchain_data)
        return anomalies

cpsbs = CPSBS()
