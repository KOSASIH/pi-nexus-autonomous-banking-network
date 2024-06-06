# agi_decision_making.py 
import numpy as np
from artificial_general_intelligence import ArtificialGeneralIntelligence

class AGIDM:
    def __init__(self):
        self.agi = ArtificialGeneralIntelligence()

    def make_decision(self, blockchain_data):
        decision = self.agi.make_decision(blockchain_data)
        return decision

    def explain_decision(self, blockchain_data):
        explanation = self.agi.explain_decision(blockchain_data)
        return explanation

agidm = AGIDM()
