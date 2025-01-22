# deutsch_jozsa.py
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def deutsch_jozsa_oracle(n, constant=True):
    """
    Create the Deutsch-Jozsa oracle.
    
    Parameters:
    - n: Number of input qubits
    - constant: If True, the function is constant; if False, it is balanced.
    
    Returns:
    - QuantumCircuit: The oracle circuit
    """
    oracle = QuantumCircuit(n + 1)  # n input qubits + 1 output qubit

    if constant:
        # If the function is constant, do nothing (output is always 0)
        pass
    else:
        # If the function is balanced, flip the output for half of the inputs
        for i in range(2**(n-1)):
            binary_input = format(i, f'0{n}b')
            oracle.x(n)  # Set the output qubit to |1>
            for j in range(n):
                if binary_input[j] == '1':
                    oracle.x(j)  # Flip the input qubits to match the balanced condition
            oracle.mct(list(range(n)), n)  # Multi-controlled Toffoli gate
            for j in range(n):
                if binary_input[j] == '1':
                    oracle.x(j)  # Revert the input qubits back

    return oracle

def deutsch_jozsa_algorithm(n, constant=True):
    """
    Implement the Deutsch-Jozsa algorithm.
    
    Parameters:
    - n: Number of input qubits
    - constant: If True, the function is constant; if False, it is balanced.
    
    Returns:
    - Result of the measurement
    """
    # Create a quantum circuit with n input qubits and 1 output qubit
    circuit = QuantumCircuit(n + 1, n)

    # Initialize the output qubit to |1>
    circuit.x(n)  # Set the output qubit to |1>
    circuit.h(range(n + 1))  # Apply Hadamard gates to all qubits

    # Create the oracle
    oracle = deutsch_jozsa_oracle(n, constant)
    circuit.append(oracle, range(n + 1))

    # Apply Hadamard gates to the input qubits again
    circuit.h(range(n))

    # Measure the input qubits
    circuit.measure(range(n), range(n))

    return circuit

def run_deutsch_jozsa(n, constant=True):
    """
    Run the Deutsch-Jozsa algorithm and return the results.
    
    Parameters:
    - n: Number of input qubits
    - constant: If True, the function is constant; if False, it is balanced.
    """
    # Create and run the Deutsch-Jozsa circuit
    circuit = deutsch_jozsa_algorithm(n, constant)

    # Use the Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1024)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)

    return counts

if __name__ == "__main__":
    n = 3  # Number of input qubits
    constant = False  # Change to True for a constant function
    counts = run_deutsch_jozsa(n, constant)

    # Print the results
    print("Counts:", counts)

    # Plot the results
    plot_histogram(counts).show()
