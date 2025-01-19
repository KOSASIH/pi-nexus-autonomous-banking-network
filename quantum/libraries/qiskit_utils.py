# qiskit_utils.py
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
    circuit.h(range(num_qubits))  # Apply Hadamard to all qubits
    circuit.measure(range(num_qubits), range(num_qubits))  # Measure all qubits
    return circuit

def create_entangled_circuit(num_qubits):
    """
    Create a quantum circuit that generates a Bell state (entangled state).
    
    Parameters:
    - num_qubits: Number of qubits (must be at least 2)
    
    Returns:
    - QuantumCircuit: The constructed entangled circuit
    """
    if num_qubits < 2:
        raise ValueError("At least 2 qubits are required for entanglement.")
    
    circuit = QuantumCircuit(num_qubits, 2)
    circuit.h(0)  # Apply Hadamard to the first qubit
    circuit.cx(0, 1)  # Apply CNOT to create entanglement
    circuit.measure(0, 0)
    circuit.measure(1, 1)
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
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1024)
    result = job.result()
    
    counts = result.get_counts(circuit)
    statevector = Statevector.from_dict(counts)
    
    return counts, statevector

def visualize_results(counts, statevector):
    """
    Visualize the results of the quantum circuit execution.
    
    Parameters:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    print("Counts:", counts)
    plot_histogram(counts).show()
    plot_bloch_multivector(statevector).show()

def prepare_state(circuit, state_vector):
    """
    Prepare a quantum state from a given state vector.
    
    Parameters:
    - circuit: QuantumCircuit object
    - state_vector: List or array representing the quantum state
    
    Returns:
    - QuantumCircuit: The updated circuit with the prepared state
    """
    num_qubits = int(np.log2(len(state_vector)))
    circuit.initialize(state_vector, range(num_qubits))
    return circuit

def apply_noise(circuit, noise_model):
    """
    Apply a noise model to the quantum circuit.
    
    Parameters:
    - circuit: QuantumCircuit object
    - noise_model: NoiseModel object to apply
    
    Returns:
    - QuantumCircuit: The circuit with noise applied
    """
    from qiskit.providers.aer.noise import NoiseModel
    return transpile(circuit, noise_model=noise_model)

if __name__ == "__main__":
    # Example usage of the utility functions
    num_qubits = 2
    entangled_circuit = create_entangled_circuit(num_qubits)
    counts, statevector = run_circuit(entangled_circuit)
    visualize_results(counts, statevector)
