# quantum_metrics.py
import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix

def fidelity(state1, state2):
    """
    Calculate the fidelity between two quantum states.
    
    Parameters:
    - state1: First quantum state (Statevector or DensityMatrix)
    - state2: Second quantum state (Statevector or DensityMatrix)
    
    Returns:
    - float: Fidelity value between 0 and 1
    """
    if isinstance(state1, Statevector) and isinstance(state2, Statevector):
        return np.abs(np.dot(state1, state2.conjugate()))**2
    elif isinstance(state1, DensityMatrix) and isinstance(state2, DensityMatrix):
        return np.trace(np.sqrt(state1.sqrt() @ state2 @ state1.sqrt()))**2
    else:
        raise ValueError("Both states must be either Statevector or DensityMatrix.")

def success_probability(counts, target_state):
    """
    Calculate the success probability of measuring a target state.
    
    Parameters:
    - counts: Measurement results (dict)
    - target_state: The target state to measure (str)
    
    Returns:
    - float: Success probability of measuring the target state
    """
    total_shots = sum(counts.values())
    target_count = counts.get(target_state, 0)
    return target_count / total_shots if total_shots > 0 else 0.0

def average_execution_time(times):
    """
    Calculate the average execution time of quantum circuits.
    
    Parameters:
    - times: List of execution times (in seconds)
    
    Returns:
    - float: Average execution time
    """
    return np.mean(times) if times else 0.0

def analyze_results(counts, target_state, execution_times):
    """
    Analyze the results of a quantum algorithm.
    
    Parameters:
    - counts: Measurement results (dict)
    - target_state: The target state to measure (str)
    - execution_times: List of execution times (in seconds)
    
    Returns:
    - dict: Dictionary containing fidelity, success probability, and average execution time
    """
    results = {}
    results['success_probability'] = success_probability(counts, target_state)
    results['average_execution_time'] = average_execution_time(execution_times)
    return results

if __name__ == "__main__":
    # Example usage of the quantum metrics functions
    # Simulated measurement results
    counts = {'00': 500, '01': 300, '10': 200, '11': 100}
    target_state = '00'
    execution_times = [0.1, 0.15, 0.12, 0.14]  # Example execution times in seconds

    # Analyze results
    results = analyze_results(counts, target_state, execution_times)
    print("Analysis Results:")
    print(f"Success Probability: {results['success_probability']:.2f}")
    print(f"Average Execution Time: {results['average_execution_time']:.2f} seconds")
