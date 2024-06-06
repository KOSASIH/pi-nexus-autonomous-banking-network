# sibc_consensus.py
import numpy as np
from swarm_intelligence import SwarmIntelligence

class SIBC:
    def __init__(self):
        self.si = SwarmIntelligence()

    def consensus(self, nodes):
        consensus = self.si.consensus(nodes)
        return consensus

    def adapt_to_network_topology(self, nodes, network_topology):
        adapted_consensus = self.si.adapt(nodes, network_topology)
        return adapted_consensus

sibc = SIBC()
