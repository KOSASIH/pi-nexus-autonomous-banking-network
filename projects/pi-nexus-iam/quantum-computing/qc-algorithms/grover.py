import numpy as np
from qiskit import QuantumCircuit, execute, Aer

def grover_oracle(f, n: int) -> QuantumCircuit:
    """
    Creates a Grover oracle circuit for the function f.

    Args:
    f: The function to be searched.
    n: The number of qubits.

    Returns:
    The Grover oracle circuit.
    """
    qc = QuantumCircuit(n + 1)

    # Apply the function f
    for i in range(n):
        if f(i):
            qc.x(n)

    # Apply the diffusion operator
    qc.h(range(n))
    qc.x(range(n))
    qc.h(range(n))

    return qc

def grover_search(f, n: int) -> int:
    """
    Grover's algorithm for searching an unsorted database.

    Args:
    f: The function to be searched.
    n: The number of qubits.

    Returns:
    The index of the marked element.
    """
    qc = QuantumCircuit(n + 1)

    # Initialize the qubits
    qc.h(range(n))

    # Apply the Grover oracle
    for _ in range(int(np.pi / 4) * np.sqrt(2 ** n)):
        qc.compose(grover_oracle(f, n), inplace=True)

    # Measure the qubits
    qc.measure(range(n), range(n))

    # Run the circuit on a simulator
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1024)
    result = job.result()

    # Extract the index from the measurement results
    index = 0
    for i, count in enumerate(result.get_counts().items()):
        if count[1] > 100:
            index = i
            break

    return index

# Example usage
def f(x: int) -> bool:
    return x == 5

n = 6
index = grover_search(f, n)
print(f"Index of the marked element: {index}")
