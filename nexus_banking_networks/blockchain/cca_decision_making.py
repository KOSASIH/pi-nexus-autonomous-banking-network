# cca_decision_making.py
import numpy as np
from cybernetic_cognitive_architecture import CyberneticCognitiveArchitecture

class CCADM:
    def __init__(self):
        self.cca = CyberneticCognitiveArchitecture()

    def make_decision(self, blockchain_data):
        decision = self.cca.decide(blockchain_data)
        return decision

ccadm = CCADM()
