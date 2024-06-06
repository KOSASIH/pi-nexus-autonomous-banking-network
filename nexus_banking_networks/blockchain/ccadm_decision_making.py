# ccadm_decision_making.py
import numpy as np
from cybernetic_cognitive_architecture import CyberneticCognitiveArchitecture

class CCADM:
    def __init__(self):
        self.cca = CyberneticCognitiveArchitecture()

    def make_decision(self, blockchain_data):
        decision = self.cca.decide(blockchain_data)
        return decision

    def explain_decision(self, blockchain_data):
        explanation = self.cca.explain(blockchain_data)
        return explanation

ccadm = CCADM()
