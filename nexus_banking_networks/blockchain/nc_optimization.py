# nc_optimization.py
import numpy as np
from neuromorphic_computing import NeuromorphicComputing

class NCO:
    def __init__(self):
        self.nc = NeuromorphicComputing()

    def optimize_blockchain(self, blockchain_data):
        optimized_data = self.nc.optimize(blockchain_data)
        return optimized_data

    def explain_optimization(self, blockchain_data):
        explanation = self.nc.explain(blockchain_data)
        return explanation

nco = NCO()
