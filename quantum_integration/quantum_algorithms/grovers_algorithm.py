from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def grovers_algorithm(n, target):
    # Create a quantum circuit with n qubits
    qc = QuantumCircuit(n, n)
    
    # Initialize the target state
    qc.h(range(n))  # Apply Hadamard to all qubits
    qc.x(target)    # Flip the target qubit
    qc.h(target)    # Apply Hadamard to the target qubit
    
    # Oracle for the target state
    qc = oracle(qc, n, target)
    
    # Grover's diffusion operator
    qc.h(range(n))
    qc.x(range(n))
    qc.h(target)
    qc.x(target)
    qc.h(range(n))
    
    # Measure the qubits
    qc.measure(range(n), range(n))
    
    # Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1024).result()
    counts = result.get_counts()
    
    # Plot the results
    plot_histogram(counts).show()
    
    return counts

def oracle(qc, n, target):
    # Implement the oracle for the target state
    for qubit in range(n):
        if qubit != target:
            qc.x(qubit)  # Flip all qubits except the target
    qc.h(target)
    qc.mct(list(range(n)), target)  # Multi-controlled Toffoli gate
    qc.h(target)
    for qubit in range(n):
        if qubit != target:
            qc.x(qubit)  # Flip back the qubits
    return qc
