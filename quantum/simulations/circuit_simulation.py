# circuit_simulation.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector

def create_quantum_circuit(num_qubits, gates):
    """
    Create a quantum circuit with specified gates.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - gates: List of gates to apply (e.g., ['h', 'cx', 'x'])
    
    Returns:
    - QuantumCircuit: The constructed quantum circuit
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)  # Create circuit with classical bits for measurement

    # Apply specified gates
    for gate in gates:
        if gate == 'h':
            for qubit in range(num_qubits):
                circuit.h(qubit)  # Apply Hadamard to all qubits
        elif gate == 'cx':
            if num_qubits > 1:
                circuit.cx(0, 1)  # Apply CNOT between first two qubits
        elif gate == 'x':
            circuit.x(0)  # Apply X gate to the first qubit
        elif gate == 'y':
            circuit.y(0)  # Apply Y gate to the first qubit
        elif gate == 'z':
            circuit.z(0)  # Apply Z gate to the first qubit

    # Measure the qubits
    circuit.measure(range(num_qubits), range(num_qubits))

    return circuit

def run_circuit_simulation(num_qubits, gates):
    """
    Run the quantum circuit simulation and return the results.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - gates: List of gates to apply
    
    Returns:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    # Create the quantum circuit
    circuit = create_quantum_circuit(num_qubits, gates)

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
    Visualize the results of the circuit simulation.
    
    Parameters:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    print("Counts:", counts)
    plot_histogram(counts).show()
    plot_bloch_multivector(statevector).show()

if __name__ == "__main__":
    num_qubits = 2  # Number of qubits to simulate
    gates = ['h', 'cx']  # List of gates to apply

    # Run the circuit simulation
    counts, statevector = run_circuit_simulation(num_qubits, gates)

    # Visualize the results
    visualize_results(counts, statevector)
