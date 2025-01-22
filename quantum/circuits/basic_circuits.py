# basic_circuits.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector

def create_hadamard_circuit(num_qubits):
    """
    Create a quantum circuit with Hadamard gates applied to all qubits.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    
    Returns:
    - QuantumCircuit: The constructed quantum circuit
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)
    for qubit in range(num_qubits):
        circuit.h(qubit)  # Apply Hadamard gate
    circuit.measure(range(num_qubits), range(num_qubits))  # Measure all qubits
    return circuit

def create_cnot_circuit(num_qubits):
    """
    Create a quantum circuit with a CNOT gate applied between the first two qubits.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    
    Returns:
    - QuantumCircuit: The constructed quantum circuit
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)
    if num_qubits > 1:
        circuit.cx(0, 1)  # Apply CNOT gate
    circuit.measure(range(num_qubits), range(num_qubits))  # Measure all qubits
    return circuit

def create_entanglement_circuit():
    """
    Create a quantum circuit that generates a Bell state (entangled state).
    
    Returns:
    - QuantumCircuit: The constructed quantum circuit
    """
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)  # Apply Hadamard to the first qubit
    circuit.cx(0, 1)  # Apply CNOT to create entanglement
    circuit.measure(range(2), range(2))  # Measure both qubits
    return circuit

def run_circuit(circuit):
    """
    Run the quantum circuit and return the results.
    
    Parameters:
    - circuit: QuantumCircuit object
    
    Returns:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    # Use the Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1024)
    result = job.result()

    # Get measurement counts
    counts = result.get_counts(circuit)

    # Get the state vector for visualization
    statevector = Statevector.from_dict(counts)

    return counts, statevector

def visualize_results(counts, statevector):
    """
    Visualize the results of the quantum circuit simulation.
    
    Parameters:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    print("Counts:", counts)
    plot_histogram(counts).show()
    plot_bloch_multivector(statevector).show()

if __name__ == "__main__":
    # Example usage of the circuit functions
    num_qubits = 2  # Number of qubits

    # Create and run Hadamard circuit
    hadamard_circuit = create_hadamard_circuit(num_qubits)
    counts, statevector = run_circuit(hadamard_circuit)
    print("Hadamard Circuit Results:")
    visualize_results(counts, statevector)

    # Create and run CNOT circuit
    cnot_circuit = create_cnot_circuit(num_qubits)
    counts, statevector = run_circuit(cnot_circuit)
    print("CNOT Circuit Results:")
    visualize_results(counts, statevector)

    # Create and run entanglement circuit
    entanglement_circuit = create_entanglement_circuit()
    counts, statevector = run_circuit(entanglement_circuit)
    print("Entanglement Circuit Results:")
    visualize_results(counts, statevector)
