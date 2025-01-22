# q_simulation.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
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
    circuit = QuantumCircuit(num_qubits, num_qubits)

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

def add_noise_model(circuit, noise_level=0.1):
    """
    Add noise to the quantum circuit.
    
    Parameters:
    - circuit: QuantumCircuit object
    - noise_level: Level of noise to apply (0 to 1)
    
    Returns:
    - NoiseModel: The noise model applied to the circuit
    """
    noise_model = NoiseModel()
    error = depolarizing_error(noise_level, 1)  # 1-qubit depolarizing error
    noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'y', 'z', 'cx'])
    return noise_model

def run_simulation(num_qubits, gates, noise_level=0.0):
    """
    Run the quantum simulation and return the results.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - gates: List of gates to apply
    - noise_level: Level of noise to apply (0 to 1)
    
    Returns:
    - Counts of the measurement results
    """
    # Create the quantum circuit
    circuit = create_quantum_circuit(num_qubits, gates)

    # Add noise if specified
    noise_model = add_noise_model(circuit, noise_level) if noise_level > 0 else None

    # Use the Aer's qasm_simulator
    simulator = AerSimulator()

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1024, noise_model=noise_model)
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)

    # Get the state vector for visualization
    statevector = Statevector.from_dict(counts)
    
    return counts, statevector

def visualize_results(counts, statevector):
    """
    Visualize the results of the simulation.
    
    Parameters:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    print("Counts:", counts)
    plot_histogram(counts).show()
    plot_state_qsphere(statevector).show()
    plot_bloch_multivector(statevector).show()

if __name__ == "__main__":
    num_qubits = 2  # Number of qubits to simulate
    gates = ['h', 'cx']  # List of gates to apply
    noise_level = 0.1  # Level of noise to apply

    # Run the simulation
    counts, statevector = run_simulation(num_qubits, gates, noise_level)

    # Visualize the results
    visualize_results(counts, statevector)
