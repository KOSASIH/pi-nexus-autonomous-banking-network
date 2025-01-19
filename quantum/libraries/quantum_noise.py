# quantum_noise.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit.quantum_info import Statevector

def create_noise_model(noise_type='depolarizing', noise_level=0.1):
    """
    Create a noise model based on the specified type and level.
    
    Parameters:
    - noise_type: Type of noise ('depolarizing', 'amplitude_damping', 'phase_damping')
    - noise_level: Level of noise (0 to 1)
    
    Returns:
    - NoiseModel: The constructed noise model
    """
    noise_model = NoiseModel()
    
    if noise_type == 'depolarizing':
        error = depolarizing_error(noise_level, 1)  # 1-qubit depolarizing error
        noise_model.add_all_qubit_quantum_error(error, ['h', 'cx', 'measure'])
    elif noise_type == 'amplitude_damping':
        error = amplitude_damping_error(noise_level)
        noise_model.add_all_qubit_quantum_error(error, ['h', 'cx', 'measure'])
    elif noise_type == 'phase_damping':
        error = phase_damping_error(noise_level)
        noise_model.add_all_qubit_quantum_error(error, ['h', 'cx', 'measure'])
    else:
        raise ValueError("Unsupported noise type. Choose from 'depolarizing', 'amplitude_damping', or 'phase_damping'.")
    
    return noise_model

def apply_noise_to_circuit(circuit, noise_model):
    """
    Apply a noise model to a quantum circuit.
    
    Parameters:
    - circuit: QuantumCircuit object
    - noise_model: NoiseModel object to apply
    
    Returns:
    - QuantumCircuit: The circuit with noise applied
    """
    return circuit.decompose()  # Decompose the circuit to apply noise

def run_circuit_with_noise(circuit, noise_model):
    """
    Run the quantum circuit with the specified noise model and return the results.
    
    Parameters:
    - circuit: QuantumCircuit object
    - noise_model: NoiseModel object to apply
    
    Returns:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    # Use the Aer's qasm_simulator
    simulator = AerSimulator()

    # Execute the circuit on the qasm simulator with noise
    job = execute(circuit, simulator, shots=1024, noise_model=noise_model)
    result = job.result()

    # Get measurement counts
    counts = result.get_counts(circuit)

    # Get the state vector for visualization
    statevector = Statevector.from_dict(counts)

    return counts, statevector

def visualize_results(counts, statevector):
    """
    Visualize the results of the noise simulation.
    
    Parameters:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    print("Counts:", counts)
    plot_histogram(counts).show()
    plot_bloch_multivector(statevector).show()

if __name__ == "__main__":
    # Example usage of the noise simulation functions
    num_qubits = 2
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(0)  # Apply Hadamard to the first qubit
    circuit.cx(0, 1)  # Apply CNOT to create entanglement
    circuit.measure(range(num_qubits), range(num_qubits))  # Measure all qubits

    # Create a noise model
    noise_model = create_noise_model(noise_type='depolarizing', noise_level=0.1)

    # Run the circuit with noise
    counts, statevector = run_circuit_with_noise(circuit, noise_model)

    # Visualize the results
    visualize_results(counts, statevector)
