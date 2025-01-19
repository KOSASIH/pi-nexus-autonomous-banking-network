# quantum_feature_maps.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap

def z_feature_map(data, num_qubits):
    """
    Create a Z-Feature Map for the given data.
    
    Parameters:
    - data: np.ndarray, input data to encode (must be 1D)
    - num_qubits: int, number of qubits to use
    
    Returns:
    - QuantumCircuit: The constructed Z-Feature Map circuit
    """
    if data.ndim != 1 or len(data) != num_qubits:
        raise ValueError("Data must be a 1D array with length equal to num_qubits.")
    
    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
    circuit = feature_map.bind_parameters(data)
    
    return circuit

def pauli_feature_map(data, num_qubits):
    """
    Create a Pauli Feature Map for the given data.
    
    Parameters:
    - data: np.ndarray, input data to encode (must be 1D)
    - num_qubits: int, number of qubits to use
    
    Returns:
    - QuantumCircuit: The constructed Pauli Feature Map circuit
    """
    if data.ndim != 1 or len(data) != num_qubits:
        raise ValueError("Data must be a 1D array with length equal to num_qubits.")
    
    feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=1)
    circuit = feature_map.bind_parameters(data)
    
    return circuit

def create_feature_map(data, num_qubits, feature_map_type='z'):
    """
    Create a feature map based on the specified type.
    
    Parameters:
    - data: np.ndarray, input data to encode (must be 1D)
    - num_qubits: int, number of qubits to use
    - feature_map_type: str, type of feature map ('z' for ZFeatureMap, 'pauli' for PauliFeatureMap)
    
    Returns:
    - QuantumCircuit: The constructed feature map circuit
    """
    if feature_map_type == 'z':
        return z_feature_map(data, num_qubits)
    elif feature_map_type == 'pauli':
        return pauli_feature_map(data, num_qubits)
    else:
        raise ValueError("Unsupported feature map type. Choose 'z' or 'pauli'.")

if __name__ == "__main__":
    # Example usage of the feature map functions
    num_qubits = 3
    data = np.array([0.5, 0.2, 0.8])  # Example data

    # Create a Z-Feature Map
    z_circuit = create_feature_map(data, num_qubits, feature_map_type='z')
    print("Z-Feature Map Circuit:")
    print(z_circuit)

    # Create a Pauli Feature Map
    pauli_circuit = create_feature_map(data, num_qubits, feature_map_type='pauli')
    print("Pauli Feature Map Circuit:")
    print(pauli_circuit)
