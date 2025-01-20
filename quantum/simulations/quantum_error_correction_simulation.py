# quantum_error_correction_simulation.py
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, DensityMatrix

def shor_code_circuit():
    """
    Create a quantum circuit for the Shor error correction code.

    Returns:
    - QuantumCircuit: The constructed Shor code circuit.
    """
    circuit = QuantumCircuit(9, 3)

    # Encode the logical |0⟩ state
    circuit.h(0)  # Prepare |+⟩ state
    circuit.cx(0, 1)
    circuit.cx(0, 2)

    # Apply the first round of error correction
    circuit.barrier()
    circuit.cx(1, 3)
    circuit.cx(2, 4)
    circuit.cx(1, 5)
    circuit.cx(2, 6)

    # Measure the first qubit
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    circuit.measure(2, 2)

    return circuit

def surface_code_circuit():
    """
    Create a simple quantum circuit for the surface code.

    Returns:
    - QuantumCircuit: The constructed surface code circuit.
    """
    circuit = QuantumCircuit(9, 3)

    # Prepare the logical |0⟩ state
    circuit.h(0)
    circuit.h(1)
    circuit.h(3)
    circuit.h(4)

    # Apply CNOTs to create entanglement
    circuit.cx(0, 3)
    circuit.cx(1, 4)
    circuit.cx(3, 6)
    circuit.cx(4, 7)

    # Measure the qubits
    circuit.measure(range(3), range(3))

    return circuit

def simulate_error_correction(circuit):
    """
    Simulate the quantum error correction circuit.

    Parameters:
    - circuit: The quantum circuit to simulate.

    Returns:
    - counts: The measurement results.
    """
    # Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, backend)
    qobj = assemble(transpiled_circuit)
    result = execute(qobj, backend, shots=1024).result()

    return result.get_counts()

if __name__ == "__main__":
    # Simulate Shor code
    print("Simulating Shor Code:")
    shor_circ = shor_code_circuit()
    shor_counts = simulate_error_correction(shor_circ)
    print("Shor Code Measurement Results:")
    print(shor_counts)
    plot_histogram(shor_counts).show()

    # Simulate Surface code
    print("Simulating Surface Code:")
    surface_circ = surface_code_circuit()
    surface_counts = simulate_error_correction(surface_circ)
    print("Surface Code Measurement Results:")
    print(surface_counts)
    plot_histogram(surface_counts).show()
