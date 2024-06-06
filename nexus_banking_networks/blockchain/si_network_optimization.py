# si_network_optimization.py
import numpy as np
from swarm_intelligence import SwarmIntelligence

class SINO:
    def __init__(self):
        self.si = SwarmIntelligence()

    def optimize_network(self, network_data):
        optimized_data = self.si.optimize(network_data)
        return optimized_data

    def explain_optimization(self, network_data):
        explanation = self.si.explain(network_data)
        return explanation

sino = SINO()
