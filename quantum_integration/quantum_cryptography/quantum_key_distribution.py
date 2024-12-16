from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

def quantum_key_distribution(num_bits=10):
    # Step 1: Alice prepares a random bit string and random bases
    alice_bits = np.random.randint(2, size=num_bits)  # Random bits (0 or 1)
    alice_bases = np.random.randint(2, size=num_bits)  # Random bases (0 or 1)
    
    # Step 2: Prepare quantum states based on Alice's bits and bases
    qc = QuantumCircuit(num_bits, num_bits)
    
    for i in range(num_bits):
        if alice_bases[i] == 0:  # Z-basis
            if alice_bits[i] == 1:
                qc.x(i)  # Prepare |1> state
        else:  # X-basis
            if alice_bits[i] == 1:
                qc.h(i)  # Prepare |+> state
                qc.x(i)  # Prepare |1> state in X-basis
    
    # Step 3: Measure the qubits
    qc.measure(range(num_bits), range(num_bits))
    
    # Step 4: Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1024).result()
    counts = result.get_counts()
    
    # Step 5: Plot the results
    plot_histogram(counts).show()
    
    # Step 6: Return Alice's bits and bases for further processing
    return alice_bits, alice_bases, counts

if __name__ == "__main__":
    bits, bases, counts = quantum_key_distribution()
    print("Alice's Bits:", bits)
    print("Alice's Bases:", bases)
    print("Measurement Results:", counts)
