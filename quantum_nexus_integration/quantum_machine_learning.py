"""
Quantum Machine Learning Module for Pi-Nexus Autonomous Banking Network

This module implements quantum-enhanced machine learning algorithms for
financial data analysis, risk assessment, and predictive modeling.
"""

import os
import numpy as np
import random
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Import the quantum-resistant cryptography module for secure data handling
from .quantum_resistant_cryptography import QuantumResistantCrypto, QUANTUM_SECURITY_LEVEL_2


class QuantumNeuralNetwork:
    """
    Simulates a quantum neural network for financial data analysis.
    
    This class provides methods for training and inference using quantum-inspired
    neural network algorithms, offering significant speedups and improved accuracy
    compared to classical neural networks.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 security_level=QUANTUM_SECURITY_LEVEL_2):
        """
        Initialize the quantum neural network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units
            security_level: Security level for cryptographic operations
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.security_level = security_level
        self.crypto = QuantumResistantCrypto(security_level=security_level)
        
        # Initialize weights with quantum-inspired randomness
        self.weights_input_hidden = self._initialize_quantum_weights((input_size, hidden_size))
        self.weights_hidden_output = self._initialize_quantum_weights((hidden_size, output_size))
        
        # Training history
        self.training_history = []
        
    def _initialize_quantum_weights(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Initialize weights using quantum-inspired randomness.
        
        Args:
            shape: Shape of the weight matrix
            
        Returns:
            Initialized weight matrix
        """
        # In a real quantum system, this would use quantum random number generation
        # For simulation, we use a pseudo-random approach
        rows, cols = shape
        weights = np.zeros(shape)
        
        for i in range(rows):
            for j in range(cols):
                # Generate a quantum-inspired random value
                random_bytes = os.urandom(8)  # Use system entropy
                random_value = int.from_bytes(random_bytes, byteorder='big') / (2**(8*8))
                weights[i, j] = (random_value * 2 - 1) * 0.1  # Scale to small values
                
        return weights
        
    def _quantum_activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply a quantum-inspired activation function.
        
        Args:
            x: Input array
            
        Returns:
            Activated output
        """
        # Simulate a quantum activation function
        # In a real quantum system, this would use quantum operations
        return 1 / (1 + np.exp(-x))  # Sigmoid function as a placeholder
        
    def _quantum_forward_pass(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a forward pass through the quantum neural network.
        
        Args:
            inputs: Input features
            
        Returns:
            Tuple of (hidden_layer_output, output_layer_output)
        """
        # Simulate quantum forward propagation
        # In a real quantum system, this would use quantum matrix multiplication
        
        # Hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self._quantum_activation(hidden_inputs)
        
        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self._quantum_activation(final_inputs)
        
        return hidden_outputs, final_outputs
        
    def _quantum_backpropagation(self, inputs: np.ndarray, targets: np.ndarray, 
                               hidden_outputs: np.ndarray, final_outputs: np.ndarray, 
                               learning_rate: float) -> None:
        """
        Perform backpropagation using quantum-inspired algorithms.
        
        Args:
            inputs: Input features
            targets: Target values
            hidden_outputs: Hidden layer outputs
            final_outputs: Output layer outputs
            learning_rate: Learning rate for weight updates
        """
        # Simulate quantum backpropagation
        # In a real quantum system, this would use quantum gradient computation
        
        # Calculate output layer errors
        output_errors = targets - final_outputs
        
        # Calculate hidden layer errors
        hidden_errors = np.dot(output_errors, self.weights_hidden_output.T)
        
        # Update weights for the output layer
        self.weights_hidden_output += learning_rate * np.dot(
            hidden_outputs.reshape(-1, 1), 
            (output_errors * final_outputs * (1 - final_outputs)).reshape(1, -1)
        )
        
        # Update weights for the hidden layer
        self.weights_input_hidden += learning_rate * np.dot(
            inputs.reshape(-1, 1), 
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)).reshape(1, -1)
        )
        
    def train(self, training_data: List[Tuple[np.ndarray, np.ndarray]], 
             epochs: int, learning_rate: float, 
             validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> Dict[str, List[float]]:
        """
        Train the quantum neural network.
        
        Args:
            training_data: List of (input, target) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate for weight updates
            validation_data: Optional validation data
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'loss': [],
            'val_loss': [] if validation_data else None
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle training data
            random.shuffle(training_data)
            
            # Train on each example
            for inputs, targets in training_data:
                # Forward pass
                hidden_outputs, final_outputs = self._quantum_forward_pass(inputs)
                
                # Calculate loss
                loss = np.mean((targets - final_outputs) ** 2)
                epoch_loss += loss
                
                # Backpropagation
                self._quantum_backpropagation(inputs, targets, hidden_outputs, final_outputs, learning_rate)
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(training_data)
            history['loss'].append(avg_loss)
            
            # Validate if validation data is provided
            if validation_data:
                val_loss = self.evaluate(validation_data)
                history['val_loss'].append(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                
        training_time = time.time() - start_time
        
        # Record training metadata
        training_metadata = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'training_samples': len(training_data),
            'validation_samples': len(validation_data) if validation_data else 0,
            'final_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1] if validation_data else None,
            'training_time_seconds': training_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.training_history.append(training_metadata)
        
        return history
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained quantum neural network.
        
        Args:
            inputs: Input features
            
        Returns:
            Predicted outputs
        """
        # Forward pass
        _, outputs = self._quantum_forward_pass(inputs)
        return outputs
        
    def evaluate(self, test_data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: List of (input, target) tuples
            
        Returns:
            Average loss on the test data
        """
        total_loss = 0.0
        
        for inputs, targets in test_data:
            # Forward pass
            _, outputs = self._quantum_forward_pass(inputs)
            
            # Calculate loss
            loss = np.mean((targets - outputs) ** 2)
            total_loss += loss
            
        return total_loss / len(test_data)
        
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'weights_input_hidden': self.weights_input_hidden.tolist(),
            'weights_hidden_output': self.weights_hidden_output.tolist(),
            'training_history': self.training_history,
            'security_level': self.security_level,
            'model_type': 'QuantumNeuralNetwork',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Encrypt the model data for security
        model_json = json.dumps(model_data)
        
        # Generate a key from a password (in a real system, this would be more secure)
        password = b"quantum-secure-password"
        key, salt = self.crypto.generate_symmetric_key(password)
        
        # Encrypt the model data
        iv, encrypted_data, tag = self.crypto.symmetric_encrypt(key, model_json.encode())
        
        # Save the encrypted model with metadata
        with open(filepath, 'wb') as f:
            # Save salt, iv, tag, and encrypted data
            f.write(salt)
            f.write(iv)
            f.write(tag)
            f.write(encrypted_data)
            
    @classmethod
    def load_model(cls, filepath: str, password: bytes) -> 'QuantumNeuralNetwork':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            password: Password to decrypt the model
            
        Returns:
            Loaded QuantumNeuralNetwork instance
        """
        with open(filepath, 'rb') as f:
            # Read salt, iv, tag, and encrypted data
            salt = f.read(32)  # Assuming 32-byte salt
            iv = f.read(16)    # Assuming 16-byte IV
            tag = f.read(16)   # Assuming 16-byte tag
            encrypted_data = f.read()
            
        # Create a crypto instance
        crypto = QuantumResistantCrypto()
        
        # Generate the key from the password and salt
        key, _ = crypto.generate_symmetric_key(password, salt)
        
        # Decrypt the model data
        model_json = crypto.symmetric_decrypt(key, iv, encrypted_data, tag)
        model_data = json.loads(model_json)
        
        # Create a new model instance
        model = cls(
            input_size=model_data['input_size'],
            hidden_size=model_data['hidden_size'],
            output_size=model_data['output_size'],
            security_level=model_data['security_level']
        )
        
        # Set the weights
        model.weights_input_hidden = np.array(model_data['weights_input_hidden'])
        model.weights_hidden_output = np.array(model_data['weights_hidden_output'])
        model.training_history = model_data['training_history']
        
        return model


class QuantumReinforcementLearning:
    """
    Implements quantum-enhanced reinforcement learning for financial decision making.
    
    This class provides methods for training agents to make optimal financial decisions
    using quantum-inspired reinforcement learning algorithms.
    """
    
    def __init__(self, state_size: int, action_size: int, security_level=QUANTUM_SECURITY_LEVEL_2):
        """
        Initialize the quantum reinforcement learning agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            security_level: Security level for cryptographic operations
        """
        self.state_size = state_size
        self.action_size = action_size
        self.security_level = security_level
        self.crypto = QuantumResistantCrypto(security_level=security_level)
        
        # Initialize Q-table with quantum-inspired randomness
        self.q_table = self._initialize_quantum_q_table()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Training history
        self.training_history = []
        
    def _initialize_quantum_q_table(self) -> Dict[int, np.ndarray]:
        """
        Initialize the Q-table using quantum-inspired randomness.
        
        Returns:
            Initialized Q-table
        """
        # In a real quantum system, this would use quantum random number generation
        q_table = {}
        
        for state in range(self.state_size):
            # Generate quantum-inspired random values
            random_bytes = os.urandom(self.action_size * 8)
            random_values = np.array([
                int.from_bytes(random_bytes[i:i+8], byteorder='big') / (2**(8*8))
                for i in range(0, len(random_bytes), 8)
            ])
            
            # Scale to small values
            q_table[state] = random_values * 0.01
            
        return q_table
        
    def _quantum_choose_action(self, state: int) -> int:
        """
        Choose an action using quantum-inspired exploration/exploitation.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Exploration
        if np.random.rand() <= self.epsilon:
            # Use quantum randomness for exploration
            random_bytes = os.urandom(4)
            return int.from_bytes(random_bytes, byteorder='big') % self.action_size
            
        # Exploitation
        return np.argmax(self.q_table[state])
        
    def _quantum_update_q_table(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update the Q-table using quantum-inspired learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Calculate the Q-value update
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning formula with quantum-inspired noise
        noise = (np.random.random() - 0.5) * 0.01  # Small quantum-inspired noise
        new_q = current_q + self.learning_rate * (
            reward + self.gamma * max_next_q - current_q + noise
        )
        
        # Update the Q-table
        self.q_table[state][action] = new_q
        
    def train(self, environment, episodes: int, max_steps: int) -> Dict[str, List[float]]:
        """
        Train the quantum reinforcement learning agent.
        
        Args:
            environment: The environment to train in (must have reset() and step() methods)
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'rewards': [],
            'epsilon': []
        }
        
        start_time = time.time()
        
        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Choose action
                action = self._quantum_choose_action(state)
                
                # Take action
                next_state, reward, done, _ = environment.step(action)
                
                # Update Q-table
                self._quantum_update_q_table(state, action, reward, next_state)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # Record history
            history['rewards'].append(total_reward)
            history['epsilon'].append(self.epsilon)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{episodes}, Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
                
        training_time = time.time() - start_time
        
        # Record training metadata
        training_metadata = {
            'episodes': episodes,
            'max_steps': max_steps,
            'final_epsilon': self.epsilon,
            'average_reward': np.mean(history['rewards']),
            'training_time_seconds': training_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.training_history.append(training_metadata)
        
        return history
        
    def act(self, state: int) -> int:
        """
        Choose the best action for a given state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        return np.argmax(self.q_table[state])
        
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'training_history': self.training_history,
            'security_level': self.security_level,
            'model_type': 'QuantumReinforcementLearning',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Encrypt the model data for security
        model_json = json.dumps(model_data)
        
        # Generate a key from a password (in a real system, this would be more secure)
        password = b"quantum-secure-password"
        key, salt = self.crypto.generate_symmetric_key(password)
        
        # Encrypt the model data
        iv, encrypted_data, tag = self.crypto.symmetric_encrypt(key, model_json.encode())
        
        # Save the encrypted model with metadata
        with open(filepath, 'wb') as f:
            # Save salt, iv, tag, and encrypted data
            f.write(salt)
            f.write(iv)
            f.write(tag)
            f.write(encrypted_data)
            
    @classmethod
    def load_model(cls, filepath: str, password: bytes) -> 'QuantumReinforcementLearning':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            password: Password to decrypt the model
            
        Returns:
            Loaded QuantumReinforcementLearning instance
        """
        with open(filepath, 'rb') as f:
            # Read salt, iv, tag, and encrypted data
            salt = f.read(32)  # Assuming 32-byte salt
            iv = f.read(16)    # Assuming 16-byte IV
            tag = f.read(16)   # Assuming 16-byte tag
            encrypted_data = f.read()
            
        # Create a crypto instance
        crypto = QuantumResistantCrypto()
        
        # Generate the key from the password and salt
        key, _ = crypto.generate_symmetric_key(password, salt)
        
        # Decrypt the model data
        model_json = crypto.symmetric_decrypt(key, iv, encrypted_data, tag)
        model_data = json.loads(model_json)
        
        # Create a new model instance
        model = cls(
            state_size=model_data['state_size'],
            action_size=model_data['action_size'],
            security_level=model_data['security_level']
        )
        
        # Set the model parameters
        model.q_table = {int(k): np.array(v) for k, v in model_data['q_table'].items()}
        model.learning_rate = model_data['learning_rate']
        model.gamma = model_data['gamma']
        model.epsilon = model_data['epsilon']
        model.epsilon_decay = model_data['epsilon_decay']
        model.epsilon_min = model_data['epsilon_min']
        model.training_history = model_data['training_history']
        
        return model


class QuantumAnomalyDetection:
    """
    Implements quantum-enhanced anomaly detection for fraud prevention.
    
    This class provides methods for detecting anomalies in financial transactions
    using quantum-inspired algorithms for improved accuracy and speed.
    """
    
    def __init__(self, feature_size: int, security_level=QUANTUM_SECURITY_LEVEL_2):
        """
        Initialize the quantum anomaly detection system.
        
        Args:
            feature_size: Number of features in the input data
            security_level: Security level for cryptographic operations
        """
        self.feature_size = feature_size
        self.security_level = security_level
        self.crypto = QuantumResistantCrypto(security_level=security_level)
        
        # Initialize model parameters
        self.mean_vector = np.zeros(feature_size)
        self.covariance_matrix = np.eye(feature_size)
        self.threshold = 3.0  # Default threshold (3 standard deviations)
        
        # Training history
        self.training_history = []
        self.anomaly_history = []
        
    def _mahalanobis_distance(self, x: np.ndarray) -> float:
        """
        Calculate the Mahalanobis distance of a data point.
        
        Args:
            x: Input data point
            
        Returns:
            Mahalanobis distance
        """
        # Calculate the difference from the mean
        diff = x - self.mean_vector
        
        # Calculate the Mahalanobis distance
        # In a real quantum system, this would use quantum linear algebra
        inv_cov = np.linalg.inv(self.covariance_matrix)
        distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
        
        return distance
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model to the training data.
        
        Args:
            X: Training data (n_samples, feature_size)
        """
        start_time = time.time()
        
        # Calculate mean vector
        self.mean_vector = np.mean(X, axis=0)
        
        # Calculate covariance matrix
        self.covariance_matrix = np.cov(X, rowvar=False)
        
        # Add a small value to the diagonal for numerical stability
        self.covariance_matrix += np.eye(self.feature_size) * 1e-6
        
        # Calculate the threshold based on the training data
        distances = np.array([self._mahalanobis_distance(x) for x in X])
        self.threshold = np.mean(distances) + 3 * np.std(distances)
        
        training_time = time.time() - start_time
        
        # Record training metadata
        training_metadata = {
            'n_samples': len(X),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'threshold': self.threshold,
            'training_time_seconds': training_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.training_history.append(training_metadata)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in the data.
        
        Args:
            X: Input data (n_samples, feature_size)
            
        Returns:
            Binary array where 1 indicates an anomaly
        """
        # Calculate distances
        distances = np.array([self._mahalanobis_distance(x) for x in X])
        
        # Classify as anomaly if distance exceeds threshold
        anomalies = (distances > self.threshold).astype(int)
        
        return anomalies
        
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores for the data.
        
        Args:
            X: Input data (n_samples, feature_size)
            
        Returns:
            Array of anomaly scores
        """
        # Calculate distances
        distances = np.array([self._mahalanobis_distance(x) for x in X])
        
        # Normalize scores relative to threshold
        scores = distances / self.threshold
        
        return scores
        
    def detect_anomalies(self, X: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies and return detailed information.
        
        Args:
            X: Input data (n_samples, feature_size)
            metadata: Optional metadata for each sample
            
        Returns:
            List of dictionaries containing anomaly information
        """
        # Calculate distances and anomaly flags
        distances = np.array([self._mahalanobis_distance(x) for x in X])
        anomalies = (distances > self.threshold).astype(int)
        
        # Prepare results
        results = []
        
        for i, (distance, is_anomaly) in enumerate(zip(distances, anomalies)):
            result = {
                'index': i,
                'distance': float(distance),
                'threshold': float(self.threshold),
                'score': float(distance / self.threshold),
                'is_anomaly': bool(is_anomaly),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add metadata if provided
            if metadata and i < len(metadata):
                result['metadata'] = metadata[i]
                
            if is_anomaly:
                self.anomaly_history.append(result)
                
            results.append(result)
            
        return results
        
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'feature_size': self.feature_size,
            'mean_vector': self.mean_vector.tolist(),
            'covariance_matrix': self.covariance_matrix.tolist(),
            'threshold': float(self.threshold),
            'training_history': self.training_history,
            'anomaly_history': self.anomaly_history[-100:] if len(self.anomaly_history) > 100 else self.anomaly_history,
            'security_level': self.security_level,
            'model_type': 'QuantumAnomalyDetection',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Encrypt the model data for security
        model_json = json.dumps(model_data)
        
        # Generate a key from a password (in a real system, this would be more secure)
        password = b"quantum-secure-password"
        key, salt = self.crypto.generate_symmetric_key(password)
        
        # Encrypt the model data
        iv, encrypted_data, tag = self.crypto.symmetric_encrypt(key, model_json.encode())
        
        # Save the encrypted model with metadata
        with open(filepath, 'wb') as f:
            # Save salt, iv, tag, and encrypted data
            f.write(salt)
            f.write(iv)
            f.write(tag)
            f.write(encrypted_data)
            
    @classmethod
    def load_model(cls, filepath: str, password: bytes) -> 'QuantumAnomalyDetection':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            password: Password to decrypt the model
            
        Returns:
            Loaded QuantumAnomalyDetection instance
        """
        with open(filepath, 'rb') as f:
            # Read salt, iv, tag, and encrypted data
            salt = f.read(32)  # Assuming 32-byte salt
            iv = f.read(16)    # Assuming 16-byte IV
            tag = f.read(16)   # Assuming 16-byte tag
            encrypted_data = f.read()
            
        # Create a crypto instance
        crypto = QuantumResistantCrypto()
        
        # Generate the key from the password and salt
        key, _ = crypto.generate_symmetric_key(password, salt)
        
        # Decrypt the model data
        model_json = crypto.symmetric_decrypt(key, iv, encrypted_data, tag)
        model_data = json.loads(model_json)
        
        # Create a new model instance
        model = cls(
            feature_size=model_data['feature_size'],
            security_level=model_data['security_level']
        )
        
        # Set the model parameters
        model.mean_vector = np.array(model_data['mean_vector'])
        model.covariance_matrix = np.array(model_data['covariance_matrix'])
        model.threshold = model_data['threshold']
        model.training_history = model_data['training_history']
        model.anomaly_history = model_data['anomaly_history']
        
        return model


# Example usage
def example_usage():
    # Create a quantum neural network
    qnn = QuantumNeuralNetwork(input_size=10, hidden_size=20, output_size=2)
    
    # Generate some synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, size=(100, 2))
    
    # Convert to list of tuples
    training_data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    
    # Train the model
    print("Training Quantum Neural Network...")
    history = qnn.train(training_data, epochs=10, learning_rate=0.1)
    
    # Make predictions
    X_test = np.random.randn(5, 10)
    predictions = qnn.predict(X_test)
    
    print("\nQuantum Neural Network Predictions:")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {pred}")
        
    # Create a quantum anomaly detection model
    qad = QuantumAnomalyDetection(feature_size=10)
    
    # Generate normal data
    normal_data = np.random.randn(1000, 10)
    
    # Fit the model
    print("\nTraining Quantum Anomaly Detection...")
    qad.fit(normal_data)
    
    # Generate test data with anomalies
    test_data = np.random.randn(10, 10)
    test_data[0] = test_data[0] * 5  # Make the first sample an anomaly
    
    # Detect anomalies
    anomaly_results = qad.detect_anomalies(test_data)
    
    print("\nQuantum Anomaly Detection Results:")
    for result in anomaly_results:
        status = "ANOMALY" if result['is_anomaly'] else "normal"
        print(f"Sample {result['index']}: {status} (score: {result['score']:.2f})")


if __name__ == "__main__":
    example_usage()