import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, execute

class QCNN(nn.Module):
    def __init__(self, num_qubits, num_classes):
        super(QCNN, self).__init__()
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.qc = QuantumCircuit(num_qubits)
        self.fc = nn.Linear(2**num_qubits, num_classes)

    def forward(self, x):
        # Quantum circuit
        self.qc.h(range(self.num_qubits))
        self.qc.barrier()
        self.qc.measure(range(self.num_qubits), range(self.num_qubits))
        job = execute(self.qc, backend='qasm_simulator', shots=1024)
        result = job.result()
        counts = result.get_counts(self.qc)
        # Classical post-processing
        x = torch.tensor([counts.get('0' * self.num_qubits, 0)])
        x = F.relu(self.fc(x))
        return x

class QuantumLayer(nn.Module):
    def __init__(self, num_qubits):
        super(QuantumLayer, self).__init__()
        self.num_qubits = num_qubits
        self.qc = QuantumCircuit(num_qubits)

    def forward(self, x):
        # Quantum circuit
        self.qc.h(range(self.num_qubits))
        self.qc.barrier()
        self.qc.measure(range(self.num_qubits), range(self.num_qubits))
        job = execute(self.qc, backend='qasm_simulator', shots=1024)
        result = job.result()
        counts = result.get_counts(self.qc)
        # Classical post-processing
        x = torch.tensor([counts.get('0' * self.num_qubits, 0)])
        return x
