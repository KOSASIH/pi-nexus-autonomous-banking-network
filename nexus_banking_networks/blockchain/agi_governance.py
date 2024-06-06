# agi_governance.py
import numpy as np
from agi import ArtificialGeneralIntelligence

class AGIGovernance:
    def __init__(self):
        self.agi = ArtificialGeneralIntelligence()

    def make_decision(self, proposal):
        decision = self.agi.reason(proposal)
        return decision

    def explain_decision(self, proposal):
        explanation = self.agi.explain(proposal)
        return explanation

agi_governance = AGIGovernance()
