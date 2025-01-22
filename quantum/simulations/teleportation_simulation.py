# teleportation_simulation.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector

def create_teleportation_circuit():
    """
    Create a quantum circuit for quantum teleportation.
    
    Returns:
    - QuantumCircuit: The constructed quantum teleportation circuit
    """
    circuit = QuantumCircuit(3, 3)  # 3 qubits and 3 classical bits

    # Create entanglement between qubit 1 and qubit 2
    circuit.h(1)  # Apply Hadamard to qubit 1
    circuit.cx(1, 2)  # Apply CNOT from qubit 1 to qubit 2

    # Prepare the state to be teleported (qubit 0)
    circuit.ry(np.pi / 4, 0)  # Example state: |psi> = |+> rotated by pi/4

    # Bell measurement on qubits 0 and 1
    circuit.cx(0, 1)  # CNOT from qubit 0 to qubit 1
    circuit.h(0)      # Hadamard on qubit 0

    # Measure qubits 0 and 1
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    # Apply corrections based on measurement results
    circuit.x(2).c_if(circuit.clbits[1], 1)  # Apply X gate if measurement of qubit 1 is 1
    circuit.z(2).c_if(circuit.clbits[0], 1)  # Apply Z gate if measurement of qubit 0 is 1

    # Measure the teleported qubit (qubit 2)
    circuit.measure(2, 2)

    return circuit

def run_teleportation_simulation():
    """
    Run the quantum teleportation simulation and return the results.
    
    Returns:
    - Counts of the measurement results
    - Statevector of the quantum system
    """
    # Create the teleportation circuit
    circuit = create_teleportation_circuit()

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
    Visualize the results of the teleportation simulation.
    
    Parameters:
    - counts: Measurement results
    - statevector: State vector of the quantum system
    """
    print("Counts:", counts)
    plot_histogram(counts).show()
    plot_bloch_multivector(statevector).show()

if __name__ == "__main__":
    # Run the teleportation simulation
    counts, statevector = run_teleportation_simulation()

    # Visualize the results
    visualize_results(counts, statevector)
