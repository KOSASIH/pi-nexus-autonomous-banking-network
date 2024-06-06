# qeca_consensus.py
import numpy as np
from qiskit import QuantumCircuit, execute

class QECA:
    def __init__(self):
        self.qc = QuantumCircuit(5, 5)

    def entangle_nodes(self, nodes):
        for i in range(len(nodes)):
            self.qc.h(i)
            self.qc.cx(i, (i+1)%len(nodes))
        self.qc.barrier()

    def measure_consensus(self):
        job = execute(self.qc, backend='qasm_simulator', shots=1024)
        result = job.result()
        counts = result.get_counts(self.qc)
        consensus = max(counts, key=counts.get)
        return consensus

qeca = QECA()
