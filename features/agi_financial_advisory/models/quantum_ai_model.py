# File name: quantum_ai_model.py
import pennylane as qml
import torch
from torch import nn

class QuantumAIModel(nn.Module):
    def __init__(self, num_qubits, num_classes):
        super(QuantumAIModel, self).__init__()
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.qnn = qml.QNN(num_qubits, num_classes)

    def forward(self, x):
        x = self.qnn(x)
        x = torch.relu(x)
        x = self.fc(x)
        return x

    def fc(self, x):
        x = torch.nn.Linear(x.shape[1], self.num_classes)(x)
        return x

model = QuantumAIModel(4, 2)
