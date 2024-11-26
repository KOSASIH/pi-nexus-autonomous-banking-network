class QuantumEntanglement:
    def create_entangled_pairs(self):
        """Simulate the creation of entangled qubit pairs."""
        print("Creating entangled qubit pairs.")
        return [("Q1", "Q2"), ("Q3", "Q4")]

    def measure_entangled(self, pair):
        """Measure the state of an entangled pair."""
        print(f"Measuring entangled pair: {pair}")

# Example usage
if __name__ == '__main__':
    entanglement = QuantumEntanglement()
    pairs = entanglement.create_entangled_pairs()
    for pair in pairs:
        entanglement.measure_entangled(pair)
