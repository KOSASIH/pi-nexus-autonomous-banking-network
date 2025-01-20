# quantum_experiment_design.py
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram

def create_parameterized_circuit(params):
    """
    Create a parameterized quantum circuit.

    Parameters:
    - params: A list of parameters for the circuit.

    Returns:
    - QuantumCircuit: The constructed parameterized circuit.
    """
    circuit = QuantumCircuit(1)  # Single qubit circuit
    circuit.rx(params[0], 0)  # Apply RX rotation with the first parameter
    circuit.ry(params[1], 0)  # Apply RY rotation with the second parameter
    circuit.measure_all()  # Measure the qubit
    return circuit

def run_experiment(circuit):
    """
    Run the quantum circuit on a simulator and return the results.

    Parameters:
    - circuit: The quantum circuit to simulate.

    Returns:
    - dict: The measurement results.
    """
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, backend)
    qobj = assemble(transpiled_circuit)
    result = execute(qobj, backend, shots=1024).result()
    return result.get_counts()

def parameter_sweep(param_ranges):
    """
    Perform a parameter sweep over specified ranges.

    Parameters:
    - param_ranges: A list of tuples specifying the ranges for each parameter.

    Returns:
    - results: A dictionary of results for each parameter combination.
    """
    results = {}
    for param1 in np.linspace(param_ranges[0][0], param_ranges[0][1], num=5):
        for param2 in np.linspace(param_ranges[1][0], param_ranges[1][1], num=5):
            params = [param1, param2]
            circuit = create_parameterized_circuit(params)
            counts = run_experiment(circuit)
            results[(param1, param2)] = counts
    return results

def optimize_experiment(param_ranges, objective_function):
    """
    Optimize the parameters of the quantum circuit based on an objective function.

    Parameters:
    - param_ranges: A list of tuples specifying the ranges for each parameter.
    - objective_function: A function that takes parameters and returns a score.

    Returns:
    - best_params: The parameters that yield the best score.
    - best_score: The best score obtained.
    """
    best_score = float('-inf')
    best_params = None

    for param1 in np.linspace(param_ranges[0][0], param_ranges[0][1], num=10):
        for param2 in np.linspace(param_ranges[1][0], param_ranges[1][1], num=10):
            params = [param1, param2]
            score = objective_function(params)
            if score > best_score:
                best_score = score
                best_params = params

    return best_params, best_score

def example_objective_function(params):
    """
    Example objective function to maximize the probability of measuring |0⟩.

    Parameters:
    - params: A list of parameters for the circuit.

    Returns:
    - float: The score based on the measurement results.
    """
    circuit = create_parameterized_circuit(params)
    counts = run_experiment(circuit)
    probability_of_zero = counts.get('0', 0) / 1024  # Probability of measuring |0⟩
    return probability_of_zero

if __name__ == "__main__":
    # Example usage of parameter sweep
    param_ranges = [(0, np.pi), (0, np.pi)]  # Ranges for RX and RY parameters
    sweep_results = parameter_sweep(param_ranges)
    print("Parameter Sweep Results:")
    for params, counts in sweep_results.items():
        print(f"Params: {params}, Counts: {counts}")

    # Example usage of optimization
    best_params, best_score = optimize_experiment(param_ranges, example_objective_function)
    print(f"Best Parameters: {best_params}, Best Score: {best_score}")
