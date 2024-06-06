# ci_autonomous_agents.py
import numpy as np
from cybernetic_intelligence import CyberneticIntelligence

class CIAA:
    def __init__(self):
        self.ci = CyberneticIntelligence()

    def make_decision(self, context):
        decision = self.ci.reason(context)
        return decision

    def learn_from_experience(self, experience):
        self.ci.learn(experience)

ciaa = CIAA()
