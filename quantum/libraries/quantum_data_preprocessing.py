# quantum_data_preprocessing.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit

def normalize_data(data):
    """
    Normalize the input data to the range [0, 1].
    
    Parameters:
    - data: np.ndarray, input data to normalize
    
    Returns:
    - np.ndarray: Normalized data
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def encode_data_to_quantum_state(data):
    """
    Encode classical data into quantum states using amplitude encoding.
    
    Parameters:
    - data: np.ndarray, normalized data to encode (must be 1D)
    
    Returns:
    - QuantumCircuit: Quantum circuit that prepares the quantum state
    """
    if data.ndim != 1:
        raise ValueError("Data must be a 1D array for amplitude encoding.")
    
    # Normalize the data to ensure it sums to 1
    data = data / np.linalg.norm(data)
    
    num_qubits = int(np.ceil(np.log2(len(data))))
    circuit = QuantumCircuit(num_qubits)

    # Prepare the quantum state using amplitude encoding
    circuit.initialize(data, range(num_qubits))
    
    return circuit

def split_dataset(data, labels, test_size=0.2):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    - data: np.ndarray, input data
    - labels: np.ndarray, corresponding labels
    - test_size: float, proportion of the dataset to include in the test split
    
    Returns:
    - tuple: (X_train, X_test, y_train, y_test)
    """
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_index = int(num_samples * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train, X_test = data[train_indices], data[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage of the data preprocessing functions
    # Sample data
    data = np.array([[0.1, 0.2], [0.4, 0.5], [0.6, 0.8], [0.9, 0.1]])
    labels = np.array([0, 1, 0, 1])

    # Normalize the data
    normalized_data = normalize_data(data)
    print("Normalized Data:\n", normalized_data)

    # Encode the first sample into a quantum state
    quantum_circuit = encode_data_to_quantum_state(normalized_data[0])
    print("Quantum Circuit for Encoding:\n", quantum_circuit)

    # Split the dataset
    X_train, X_test, y_train, y_test = split_dataset(normalized_data, labels)
    print("Training Data:\n", X_train)
    print("Testing Data:\n", X_test)
    print("Training Labels:\n", y_train)
    print("Testing Labels:\n", y_test)
