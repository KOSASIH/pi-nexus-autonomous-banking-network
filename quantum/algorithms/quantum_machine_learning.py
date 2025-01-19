# quantum_machine_learning.py
import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import QSVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def generate_data(n_samples=100):
    """
    Generate a synthetic dataset using the 'moons' shape.
    
    Parameters:
    - n_samples: Number of samples to generate
    
    Returns:
    - X: Feature data
    - y: Labels
    """
    X, y = make_moons(n_samples=n_samples, noise=0.1)
    return X, y

def plot_data(X, y):
    """
    Plot the generated data.
    
    Parameters:
    - X: Feature data
    - y: Labels
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title("Generated Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def run_qsvm(X_train, y_train, X_test):
    """
    Run the Quantum Support Vector Machine (QSVM) algorithm.
    
    Parameters:
    - X_train: Training feature data
    - y_train: Training labels
    - X_test: Test feature data
    
    Returns:
    - Predictions for the test data
    """
    # Create a QSVC instance
    qsvm = QSVC(quantum_instance=QuantumInstance(Aer.get_backend('aer_simulator')))

    # Fit the model
    qsvm.fit(X_train, y_train)

    # Make predictions
    predictions = qsvm.predict(X_test)

    return predictions

if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_data(n_samples=200)
    plot_data(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run QSVM
    predictions = run_qsvm(X_train, y_train, X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy of QSVM: {accuracy:.2f}")
