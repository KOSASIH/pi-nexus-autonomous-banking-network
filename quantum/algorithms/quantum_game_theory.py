# quantum_game_theory.py
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def create_prisoners_dilemma_circuit(strategy_a, strategy_b):
    """
    Create a quantum circuit for the Prisoner's Dilemma.
    
    Parameters:
    - strategy_a: Strategy for Player A (0: Cooperate, 1: Defect)
    - strategy_b: Strategy for Player B (0: Cooperate, 1: Defect)
    
    Returns:
    - QuantumCircuit: The quantum circuit representing the game
    """
    circuit = QuantumCircuit(2, 2)

    # Initialize the qubits based on the strategies
    if strategy_a == 1:
        circuit.x(0)  # Player A defects
    if strategy_b == 1:
        circuit.x(1)  # Player B defects

    # Apply a controlled-NOT gate to simulate the interaction
    circuit.cx(0, 1)

    # Measure the qubits
    circuit.measure([0, 1], [0, 1])

    return circuit

def run_prisoners_dilemma(strategy_a, strategy_b):
    """
    Run the Prisoner's Dilemma quantum circuit and return the results.
    
    Parameters:
    - strategy_a: Strategy for Player A (0: Cooperate, 1: Defect)
    - strategy_b: Strategy for Player B (0: Cooperate, 1: Defect)
    
    Returns:
    - Counts of the measurement results
    """
    # Create the quantum circuit
    circuit = create_prisoners_dilemma_circuit(strategy_a, strategy_b)

    # Use the Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1024)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)

    return counts

if __name__ == "__main__":
    # Define strategies for Player A and Player B
    strategy_a = 0  # Player A cooperates
    strategy_b = 1  # Player B defects

    # Run the Prisoner's Dilemma
    counts = run_prisoners_dilemma(strategy_a, strategy_b)

    # Print the results
    print("Counts:", counts)

    # Plot the results
    plot_histogram(counts).show()
