import numpy as np
import random

class QuantumKeyDistribution:
    def __init__(self):
        self.basis_choices = ['Z', 'X']
        self.key = []
        self.basis = []
        self.eavesdropping_detected = False

    def generate_key(self, length):
        """Generate a random quantum key with basis choices."""
        for _ in range(length):
            bit = random.randint(0, 1)
            basis = random.choice(self.basis_choices)
            self.key.append((bit, basis))
            self.basis.append(basis)
        return self.key

    def measure(self, basis_choice):
        """Simulate measurement of the quantum key."""
        measured_key = []
        for (bit, basis) in self.key:
            if basis == basis_choice:
                measured_key.append(bit)
            else:
                measured_key.append(random.randint(0, 1))  # Random outcome
        return measured_key

    def reconcile_keys(self, measured_key, basis_choice):
        """Reconcile keys and check for eavesdropping."""
        if self.basis != basis_choice:
            self.eavesdropping_detected = True
        return [bit for bit, b in zip(measured_key, basis_choice) if b in self.basis]

# Example usage
if __name__ == '__main__':
    qkd = QuantumKeyDistribution()
    key = qkd.generate_key(10)
    print("Generated Quantum Key:", key)
    measured_key = qkd.measure(['Z', 'X'] * 5)
    reconciled_key = qkd.reconcile_keys(measured_key, ['Z', 'Z', 'X', 'X', 'Z', 'X', 'Z', 'X', 'Z', 'X'])
    print("Reconciled Key:", reconciled_key)
    print("Eavesdropping Detected:", qkd.eavesdropping_detected)
