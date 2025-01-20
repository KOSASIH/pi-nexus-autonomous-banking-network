# quantum_statistics.py
import numpy as np
import scipy.stats as stats

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given dataset.

    Parameters:
    - data: A list or numpy array of measurement results.
    - confidence: The confidence level for the interval (default is 0.95).

    Returns:
    - tuple: The lower and upper bounds of the confidence interval.
    """
    data = np.array(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    margin_of_error = std_err * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    
    return mean - margin_of_error, mean + margin_of_error

def perform_hypothesis_test(data1, data2, alpha=0.05):
    """
    Perform a two-sample t-test to compare the means of two datasets.

    Parameters:
    - data1: First dataset (list or numpy array).
    - data2: Second dataset (list or numpy array).
    - alpha: Significance level for the test (default is 0.05).

    Returns:
    - tuple: t-statistic, p-value, and whether to reject the null hypothesis.
    """
    t_stat, p_value = stats.ttest_ind(data1, data2)
    reject_null = p_value < alpha
    
    return t_stat, p_value, reject_null

def calculate_fidelity(state1, state2):
    """
    Calculate the fidelity between two quantum states.

    Parameters:
    - state1: First quantum state (as a DensityMatrix or Statevector).
    - state2: Second quantum state (as a DensityMatrix or Statevector).

    Returns:
    - float: The fidelity between the two states.
    """
    fidelity = np.abs(np.dot(state1, state2))**2
    return fidelity

if __name__ == "__main__":
    # Example usage
    # Simulated measurement results from a quantum experiment
    results_a = np.random.binomial(n=1, p=0.7, size=100)  # Simulated results for dataset A
    results_b = np.random.binomial(n=1, p=0.5, size=100)  # Simulated results for dataset B

    # Calculate confidence interval for dataset A
    ci_a = calculate_confidence_interval(results_a)
    print(f"Confidence Interval for Dataset A: {ci_a}")

    # Perform hypothesis test between dataset A and B
    t_stat, p_value, reject_null = perform_hypothesis_test(results_a, results_b)
    print(f"T-statistic: {t_stat}, P-value: {p_value}, Reject Null Hypothesis: {reject_null}")

    # Example fidelity calculation (using random states)
    state1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩ state
    state2 = np.array([1, 0])  # |0⟩ state
    fidelity = calculate_fidelity(state1, state2)
    print(f"Fidelity between states: {fidelity}")
