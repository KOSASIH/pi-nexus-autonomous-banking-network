import torch
from gan import CodeGAN
from torch.nn import Module, Sequential


class CodeGenerator:
    def __init__(self, model_path: str):
        self.model = CodeGAN.load_from_checkpoint(model_path)

    def generate_code(self, prompt: str) -> str:
        input_tensor = torch.tensor([prompt])
        output = self.model(input_tensor)
        return output.detach().numpy().decode("utf-8")
