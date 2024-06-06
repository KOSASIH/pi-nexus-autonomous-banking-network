# agibo_oracles.py
import numpy as np
from artificial_general_intelligence import ArtificialGeneralIntelligence

class AGIBO:
    def __init__(self):
        self.agi = ArtificialGeneralIntelligence()

    def create_oracle(self, oracle_code):
        self.agi.compile(oracle_code)
        self.agi.optimize()
        return self.agi.deploy()

    def query_oracle(self, oracle_address, input_data):
        output_data = self.agi.execute(oracle_address, input_data)
        return output_data

agibo = AGIBO()
