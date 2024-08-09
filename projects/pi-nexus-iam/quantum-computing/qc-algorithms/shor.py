import numpy as np
from qiskit import QuantumCircuit, execute, Aer

def shor_period_finding(N: int, a: int) -> int:
    """
    Shor's algorithm for period finding.

    Args:
    N: The number to factorize.
    a: The base for the modular exponentiation.

    Returns:
    The period of the function f(x) = a^x mod N.
    """
    # Create a quantum circuit with 2n + 3 qubits
    n = N.bit_length()
    qc = QuantumCircuit(2 * n + 3)

    # Initialize the qubits
    qc.h(range(n + 2))

    # Apply the modular exponentiation
    for i in range(n):
        qc.cu1(np.pi / (2 ** i), n + 1, i)

    # Apply the quantum Fourier transform
    qc.h(range(n + 2))
    qc.barrier()

    # Measure the qubits
    qc.measure(range(n + 2), range(n + 2))

    # Run the circuit on a simulator
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1024)
    result = job.result()

    # Extract the period from the measurement results
    period = 0
    for i, count in enumerate(result.get_counts().items()):
        if count[1] > 100:
            period = i
            break

    return period

def shor_factorize(N: int) -> int:
    """
    Shor's algorithm for factorizing a composite number.

    Args:
    N: The composite number to factorize.

    Returns:
    A non-trivial factor of N.
    """
    a = 2
    while True:
        period = shor_period_finding(N, a)
        if period % 2 == 0:
            break
        a += 1

    factor = pow(a, period // 2, N)
    if factor * factor != N:
        return factor
    else:
        return shor_factorize(N)

# Example usage
N = 56153
factor = shor_factorize(N)
print(f"Factor of {N}: {factor}")
