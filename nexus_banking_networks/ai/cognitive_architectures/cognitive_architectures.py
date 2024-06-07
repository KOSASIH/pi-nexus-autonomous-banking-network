import torch
import torch.nn as nn
from cognitive_architectures import SOAR

class CognitiveAgent:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.soar = SOAR(num_inputs, num_outputs)

    def reason(self, input_data):
        output = self.soar.reason(input_data)
        return output

class CognitiveSystem:
    def __init__(self, cognitive_agent):
        self.cognitive_agent = cognitive_agent

    def make_decision(self, input_data):
        output = self.cognitive_agent.reason(input_data)
        return output
