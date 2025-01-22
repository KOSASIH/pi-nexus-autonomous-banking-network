# noise_simulation.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Statevector

def create_noise_circuit():
    """
    Create a simple quantum circuit with a Hadamard gate and a measurement.
    
    Returns:
    - QuantumCircuit: The constructed quantum circuit
    """
    circuit = QuantumCircuit(1, 1)  # 1 qubit and 1 classical bit

    # Apply a Hadamard gate to create superposition
    circuit.h(0)

    # Measure the qubit
    circuit.measure(0, 0)

    return circuit

def add_noise_model(circuit, noise_level=0.1):
    """
    Add a depolarizing noise model to the quantum circuit.
    
    Parameters:
    - circuit: QuantumCircuit object
    - noise_level: Level of noise to apply (0 to 1)
    
    Returns:
    - NoiseModel: The noise model applied to the circuit
    """
    noise_model = NoiseModel()
    error = depolarizing_error(noise_level, 1)  # 1-qubit depolarizing error
    noise_model.add_all_qubit_quantum_error(error, ['h', 'measure'])
    return noise_model

def run_noise_simulation(noise_level=0.1):
    """
    Run the noise simulation and return the results.
    
    Parameters:
    - noise_level: Level of noise to apply (0 to 1)
    
    Returns:
    - Counts of the measurement results
    - Statevector of the quantum system
    """
    # Create the noise circuit
    circuit = create_noise_circuit()

    # Add noise to the circuit
    noise_model = add_noise_model(circuit, noise_level)

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
    noise_level = 0.1  # Level of noise to apply

    # Run the noise simulation
    counts, statevector = run_noise_simulation(noise_level)

    # Visualize the results
    visualize_results(counts, statevector)
