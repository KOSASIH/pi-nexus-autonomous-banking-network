import torch
import torch.nn as nn
from torch_explain import Explain

class ExplainableModel(nn.Module):
    def __init__(self, num_classes):
        super(ExplainableModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ExplainableAISystem:
    def __init__(self, explainable_model):
        self.explainable_model = explainable_model
        self.explain = Explain(self.explainable_model)

    def explain_prediction(self, input_data):
        output = self.explainable_model(input_data)
        explanation = self.explain.explain(output)
        return explanation
