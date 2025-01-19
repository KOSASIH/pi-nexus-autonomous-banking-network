# quantum_circuit_optimizer.py
from qiskit import QuantumCircuit
from qiskit.transpiler import transpile
from qiskit.visualization import plot_histogram
from qiskit import Aer, execute

def create_sample_circuit():
    """
    Create a sample quantum circuit for demonstration.
    
    Returns:
    - QuantumCircuit: The constructed sample circuit
    """
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)  # Apply Hadamard to qubit 0
    circuit.cx(0, 1)  # CNOT from qubit 0 to qubit 1
    circuit.cx(0, 2)  # CNOT from qubit 0 to qubit 2
    circuit.measure(range(3), range(3))  # Measure all qubits
    return circuit

def optimize_circuit(circuit):
    """
    Optimize the given quantum circuit.
    
    Parameters:
    - circuit: QuantumCircuit object to optimize
    
    Returns:
    - QuantumCircuit: The optimized circuit
    """
    # Transpile the circuit to optimize it
    optimized_circuit = transpile(circuit, optimization_level=3)  # Use the highest optimization level
    return optimized_circuit

def run_circuit(circuit):
    """
    Run the quantum circuit and return the results.
    
    Parameters:
    - circuit: QuantumCircuit object
    
    Returns:
    - counts: Measurement results
    """
    # Use the Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1024)
    result = job.result()

    # Get measurement counts
    counts = result.get_counts(circuit)
    return counts

def visualize_results(counts):
    """
    Visualize the results of the quantum circuit execution.
    
    Parameters:
    - counts: Measurement results
    """
    print("Counts:", counts)
    plot_histogram(counts).show()

if __name__ == "__main__":
    # Create a sample quantum circuit
    circuit = create_sample_circuit()
    print("Original Circuit:")
    print(circuit)

    # Optimize the circuit
    optimized_circuit = optimize_circuit(circuit)
    print("Optimized Circuit:")
    print(optimized_circuit)

    # Run the optimized circuit
    counts = run_circuit(optimized_circuit)

    # Visualize the results
    visualize_results(counts)
