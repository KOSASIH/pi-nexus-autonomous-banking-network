# qkd.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def prepare_qubit(basis, bit):
    """
    Prepare a qubit based on the chosen basis and bit value.
    
    Parameters:
    - basis: 'Z' for computational basis, 'X' for Hadamard basis
    - bit: 0 or 1
    
    Returns:
    - QuantumCircuit: A quantum circuit with the prepared qubit
    """
    qc = QuantumCircuit(1, 1)
    if basis == 'Z':
        if bit == 1:
            qc.x(0)  # Prepare |1>
    elif basis == 'X':
        if bit == 0:
            qc.h(0)  # Prepare |+>
        else:
            qc.h(0)
            qc.x(0)  # Prepare |->
    return qc

def measure_qubit(qc, basis):
    """
    Measure the qubit in the specified basis.
    
    Parameters:
    - qc: QuantumCircuit object
    - basis: 'Z' for computational basis, 'X' for Hadamard basis
    """
    if basis == 'Z':
        qc.measure(0, 0)  # Measure in computational basis
    elif basis == 'X':
        qc.h(0)  # Change to computational basis
        qc.measure(0, 0)

def bb84_protocol(num_bits):
    """
    Implement the BB84 Quantum Key Distribution protocol.
    
    Parameters:
    - num_bits: Number of bits to be transmitted
    
    Returns:
    - Alice's key, Bob's key
    """
    # Step 1: Alice prepares qubits
    alice_bits = np.random.randint(0, 2, num_bits)  # Random bits
    alice_bases = np.random.choice(['Z', 'X'], num_bits)  # Random bases
    qubits = [prepare_qubit(alice_bases[i], alice_bits[i]) for i in range(num_bits)]

    # Step 2: Bob randomly chooses bases and measures the qubits
    bob_bases = np.random.choice(['Z', 'X'], num_bits)
    bob_results = []
    
    for i in range(num_bits):
        qc = qubits[i]
        measure_qubit(qc, bob_bases[i])
        
        # Execute the circuit
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(qc, simulator, shots=1)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Get the measurement result
        bob_results.append(int(list(counts.keys())[0]))

    # Step 3: Alice and Bob announce their bases
    key_bits = []
    for i in range(num_bits):
        if alice_bases[i] == bob_bases[i]:
            key_bits.append(alice_bits[i])  # Keep the bit if bases match

    return alice_bits, bob_results, key_bits

if __name__ == "__main__":
    num_bits = 10  # Number of bits to transmit
    alice_key, bob_key, shared_key = bb84_protocol(num_bits)
    
    print("Alice's bits: ", alice_key)
    print("Bob's measurements: ", bob_key)
    print("Shared key: ", shared_key)
