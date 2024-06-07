import numpy as np
from snnlib import SNN

class NeuromorphicAgent:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.snn = SNN(num_inputs, num_outputs)

    def process(self, input_data):
        output = self.snn.process(input_data)
        return output

class RealTimeProcessor:
    def __init__(self, neuromorphic_agent):
        self.neuromorphic_agent = neuromorphic_agent

    def process_data(self, input_data):
        output = self.neuromorphic_agent.process(input_data)
        return output
