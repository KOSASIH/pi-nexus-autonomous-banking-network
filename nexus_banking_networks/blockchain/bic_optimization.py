# bic_optimization.py
import numpy as np
from bio_inspired_computing import BioInspiredComputing

class BICO:
    def __init__(self):
        self.bic = BioInspiredComputing()

    def optimize_block_size(self, input_data):
        optimized_block_size = self.bic.optimize(input_data)
        return optimized_block_size

    def explain_optimization(self, input_data):
        explanation = self.bic.explain(input_data)
        return explanation

bico = BICO()
