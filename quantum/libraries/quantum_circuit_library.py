# quantum_circuit_library.py
from qiskit import QuantumCircuit

def create_hadamard_circuit(n):
    """
    Create a circuit that applies Hadamard gates to n qubits.

    Parameters:
    - n: Number of qubits.

    Returns:
    - QuantumCircuit: The constructed circuit.
    """
    circuit = QuantumCircuit(n)
    circuit.h(range(n))  # Apply Hadamard gates to all qubits
    return circuit

def create_cnot_circuit(control, target):
    """
    Create a circuit that applies a CNOT gate from control to target qubit.

    Parameters:
    - control: The control qubit index.
    - target: The target qubit index.

    Returns:
    - QuantumCircuit: The constructed circuit.
    """
    circuit = QuantumCircuit(max(control, target) + 1)
    circuit.cx(control, target)  # Apply CNOT gate
    return circuit

def create_ghz_circuit(n):
    """
    Create a circuit that prepares a GHZ state for n qubits.

    Parameters:
    - n: Number of qubits.

    Returns:
    - QuantumCircuit: The constructed GHZ state circuit.
    """
    circuit = QuantumCircuit(n)
    circuit.h(0)  # Apply Hadamard to the first qubit
    for i in range(1, n):
        circuit.cx(0, i)  # Apply CNOT gates to create entanglement
    return circuit

def create_w_state_circuit(n):
    """
    Create a circuit that prepares a W state for n qubits.

    Parameters:
    - n: Number of qubits.

    Returns:
    - QuantumCircuit: The constructed W state circuit.
    """
    circuit = QuantumCircuit(n)
    for i in range(n - 1):
        circuit.h(i)  # Apply Hadamard to the first n-1 qubits
        circuit.cx(i, n - 1)  # Apply CNOT to the last qubit
    circuit.h(n - 1)  # Apply Hadamard to the last qubit
    return circuit

def create_measurement_circuit(n):
    """
    Create a circuit that measures n qubits.

    Parameters:
    - n: Number of qubits.

    Returns:
    - QuantumCircuit: The constructed measurement circuit.
    """
    circuit = QuantumCircuit(n, n)  # Create circuit with n classical bits
    circuit.measure(range(n), range(n))  # Measure all qubits
    return circuit

def create_entanglement_circuit():
    """
    Create a circuit that generates a Bell state (entangled state).

    Returns:
    - QuantumCircuit: The constructed Bell state circuit.
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)  # Apply Hadamard to the first qubit
    circuit.cx(0, 1)  # Apply CNOT to create entanglement
    return circuit

if __name__ == "__main__":
    # Example usage of the circuit library
    ghz_circuit = create_ghz_circuit(3)
    print("GHZ Circuit:")
    print(ghz_circuit.draw())

    w_state_circuit = create_w_state_circuit(3)
    print("\nW State Circuit:")
    print(w_state_circuit.draw())

    entanglement_circuit = create_entanglement_circuit()
    print("\nEntanglement Circuit (Bell State):")
    print(entanglement_circuit.draw())

    measurement_circuit = create_measurement_circuit(3)
    print("\nMeasurement Circuit:")
    print(measurement_circuit.draw())
