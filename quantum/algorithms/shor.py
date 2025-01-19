# shor.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from math import gcd, ceil, log2

def modular_exponentiation(base, exponent, modulus):
    """
    Perform modular exponentiation.
    
    Parameters:
    - base: The base number
    - exponent: The exponent
    - modulus: The modulus
    
    Returns:
    - Result of (base^exponent) % modulus
    """
    return pow(base, exponent, modulus)

def order_finding(a, N):
    """
    Find the order of a modulo N using quantum phase estimation.
    
    Parameters:
    - a: The base number
    - N: The number to factor
    
    Returns:
    - The order r of a modulo N
    """
    # Number of qubits needed
    n_count = ceil(log2(N))
    n = 2 * n_count

    # Create a quantum circuit
    qc = QuantumCircuit(n, n_count)

    # Initialize the counting qubits to |0>
    for qubit in range(n_count):
        qc.h(qubit)  # Apply Hadamard gates

    # Apply controlled-U operations
    for qubit in range(n_count):
        for k in range(2**qubit):
            qc.append(QFT(n_count), range(n_count))
            qc.x(qubit)
            qc.append(QFT(n_count).inverse(), range(n_count))
            qc.x(qubit)

    # Measure the counting qubits
    qc.measure(range(n_count), range(n_count))

    # Execute the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)

    # Extract the most frequent measurement result
    measured_value = max(counts, key=counts.get)
    return int(measured_value, 2)

def shors_algorithm(N):
    """
    Implement Shor's algorithm to factor N.
    
    Parameters:
    - N: The number to factor
    
    Returns:
    - A tuple of the factors of N
    """
    # Step 1: Choose a random a
    a = np.random.randint(2, N)
    
    # Step 2: Compute gcd(a, N)
    common_factor = gcd(a, N)
    if common_factor > 1:
        return common_factor, N // common_factor

    # Step 3: Find the order of a modulo N
    r = order_finding(a, N)

    # Step 4: Check if r is even and a^(r/2) is not congruent to -1 mod N
    if r % 2 != 0 or modular_exponentiation(a, r // 2, N) == N - 1:
        return None  # Failure to find factors

    # Step 5: Compute the factors
    factor1 = gcd(modular_exponentiation(a, r // 2, N) - 1, N)
    factor2 = gcd(modular_exponentiation(a, r // 2, N) + 1, N)

    return factor1, factor2

if __name__ == "__main__":
    N = 15  # Number to factor
    factors = shors_algorithm(N)
    
    if factors:
        print(f"Factors of {N} are: {factors[0]} and {factors[1]}")
    else:
        print("Failed to find factors.")
