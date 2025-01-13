from qiskit import Aer
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def quantum_svm(training_data, training_labels):
    # Create a quantum kernel
    quantum_kernel = QuantumKernel(quantum_instance=Aer.get_backend('aer_simulator'))
    
    # Create a QSVC model
    model = QSVC(quantum_kernel=quantum_kernel)
    
    # Fit the model
    model.fit(training_data, training_labels)
    
    # Predict using the model
    predictions = model.predict(training_data)
    
    return predictions

if __name__ == "__main__":
    # Generate synthetic data for classification
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the quantum SVM
    predictions = quantum_svm(X_train, y_train)
    
    # Evaluate the model
    accuracy = accuracy_score(y_train, predictions)
    print("Training Accuracy:", accuracy)
