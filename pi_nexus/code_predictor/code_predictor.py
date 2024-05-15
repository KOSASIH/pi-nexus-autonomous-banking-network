import torch
from torch.nn import Module, Sequential

class CodePredictor:
    def __init__(self, model_path: str):
        self.model = torch.load(model_path)

    def predict_code(self, input_code: str) -> str:
        input_tensor = torch.tensor([input_code])
        output = self.model(input_tensor)
        return output.detach().numpy().decode('utf-8')
