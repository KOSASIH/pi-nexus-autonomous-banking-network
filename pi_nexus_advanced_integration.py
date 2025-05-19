"""
Pi-Nexus Advanced Integration Module

This module integrates all the advanced features of the Pi-Nexus Autonomous Banking Network,
including quantum computing, AI, blockchain, biometric security, IoT, AR/VR, and more.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import quantum computing integration
from quantum_nexus_integration import (
    QuantumResistantCrypto,
    QuantumNeuralNetwork,
    QuantumReinforcementLearning,
    QuantumAnomalyDetection,
    QUANTUM_SECURITY_LEVEL_1,
    QUANTUM_SECURITY_LEVEL_2,
    QUANTUM_SECURITY_LEVEL_3
)
from quantum_nexus_integration.quantum_transaction_processor import QuantumTransactionProcessor

# Import AI integration
from ai_nexus_integration import (
    FinancialEnvironment,
    DeepQNetwork,
    FinancialDeepRLAgent
)

# Import security integration
from security_nexus_integration import (
    BiometricTemplate,
    BiometricUser,
    BiometricAuthenticator,
    DecentralizedBiometricIdentity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pi_nexus_advanced.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PiNexusAdvanced")


class PiNexusAdvancedIntegration:
    """
    Integrates all advanced features of the Pi-Nexus Autonomous Banking Network.
    
    This class serves as the main entry point for accessing all the advanced
    features of the Pi-Nexus Autonomous Banking Network, including quantum
    computing, AI, blockchain, biometric security, IoT, AR/VR, and more.
    """
    
    def __init__(self, security_level=QUANTUM_SECURITY_LEVEL_2):
        """
        Initialize the Pi-Nexus Advanced Integration.
        
        Args:
            security_level: The security level to use for cryptographic operations
        """
        logger.info("Initializing Pi-Nexus Advanced Integration")
        
        self.security_level = security_level
        self.initialization_timestamp = datetime.utcnow()
        
        # Initialize quantum computing components
        logger.info("Initializing quantum computing components")
        self.quantum_crypto = QuantumResistantCrypto(security_level=security_level)
        self.quantum_transaction_processor = QuantumTransactionProcessor(security_level=security_level)
        
        # Initialize quantum machine learning components
        logger.info("Initializing quantum machine learning components")
        self.quantum_neural_network = QuantumNeuralNetwork(
            input_size=20,
            hidden_size=40,
            output_size=10,
            security_level=security_level
        )
        self.quantum_anomaly_detection = QuantumAnomalyDetection(
            feature_size=30,
            security_level=security_level
        )
        self.quantum_reinforcement_learning = QuantumReinforcementLearning(
            state_size=100,
            action_size=20,
            security_level=security_level
        )
        
        # Initialize AI components
        logger.info("Initializing AI components")
        self.financial_environment = FinancialEnvironment(
            initial_balance=1000000.0,
            risk_tolerance=0.5,
            market_volatility=0.2,
            transaction_cost=0.001
        )
        
        # Get state size and action size for the financial environment
        state = self.financial_environment.reset()
        state_size = len(state)
        action_size = len(self.financial_environment.market_data)
        asset_names = list(self.financial_environment.market_data.keys())
        
        self.financial_rl_agent = FinancialDeepRLAgent(
            state_size=state_size,
            action_size=action_size,
            asset_names=asset_names,
            learning_rate=0.001,
            risk_tolerance=0.5
        )
        
        # Initialize security components
        logger.info("Initializing security components")
        self.biometric_authenticator = BiometricAuthenticator(security_level=security_level)
        self.decentralized_identity = DecentralizedBiometricIdentity(security_level=security_level)
        
        logger.info("Pi-Nexus Advanced Integration initialized successfully")
        
    def process_quantum_transaction(self, sender: str, recipient: str, amount: float, 
                                   currency: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a financial transaction using quantum-enhanced security.
        
        Args:
            sender: The sender's account identifier
            recipient: The recipient's account identifier
            amount: The transaction amount
            currency: The currency code (e.g., 'USD', 'EUR', 'BTC')
            metadata: Optional additional transaction metadata
            
        Returns:
            The processed transaction details
        """
        logger.info(f"Processing quantum transaction: {sender} -> {recipient}, {amount} {currency}")
        
        # Create the transaction
        transaction = self.quantum_transaction_processor.create_transaction(
            sender=sender,
            recipient=recipient,
            amount=amount,
            currency=currency,
            metadata=metadata
        )
        
        # Process the transaction
        processed_transaction = self.quantum_transaction_processor.process_transaction(transaction)
        
        logger.info(f"Quantum transaction processed: {processed_transaction['transaction_id']}, "
                   f"Status: {processed_transaction['status']}")
        
        return processed_transaction
        
    def get_optimal_investment_allocation(self, user_risk_profile: float = 0.5) -> Dict[str, float]:
        """
        Get the optimal investment allocation based on AI analysis.
        
        Args:
            user_risk_profile: The user's risk profile (0.0 to 1.0)
            
        Returns:
            A dictionary mapping asset names to allocation percentages
        """
        logger.info(f"Getting optimal investment allocation for risk profile: {user_risk_profile}")
        
        # Adjust the financial environment based on the user's risk profile
        self.financial_environment.risk_tolerance = user_risk_profile
        
        # Reset the environment to get the current state
        state = self.financial_environment.reset()
        
        # Get the optimal allocation from the RL agent
        optimal_allocation = self.financial_rl_agent.predict_optimal_allocation(state)
        
        # Filter out insignificant allocations
        significant_allocation = {asset: alloc for asset, alloc in optimal_allocation.items() if alloc > 0.01}
        
        # Normalize the significant allocations to ensure they sum to 1.0
        total = sum(significant_allocation.values())
        if total > 0:
            significant_allocation = {asset: alloc / total for asset, alloc in significant_allocation.items()}
            
        logger.info(f"Optimal investment allocation generated with {len(significant_allocation)} assets")
        
        return significant_allocation
        
    def authenticate_user(self, user_id: str, biometric_samples: Dict[str, bytes],
                         required_modalities: Optional[List[str]] = None,
                         min_modalities: int = 2) -> Dict[str, Any]:
        """
        Authenticate a user using multi-modal biometric authentication.
        
        Args:
            user_id: The ID of the user to authenticate
            biometric_samples: A dictionary mapping modalities to sample data
            required_modalities: A list of modalities that must be verified
            min_modalities: The minimum number of modalities that must be verified
            
        Returns:
            A dictionary containing authentication results
        """
        logger.info(f"Authenticating user: {user_id} with {len(biometric_samples)} biometric samples")
        
        # Perform multi-factor authentication
        success, results = self.biometric_authenticator.multi_factor_authenticate(
            user_id=user_id,
            modality_samples=biometric_samples,
            required_modalities=required_modalities,
            min_modalities=min_modalities
        )
        
        # Get the user details
        user = self.biometric_authenticator.get_user(user_id)
        
        # Prepare the authentication result
        auth_result = {
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
            'modalities': {modality: {'success': res[0], 'confidence': res[1]} for modality, res in results.items()},
            'user_info': {
                'user_id': user_id,
                'username': user.username if user else None,
                'email': user.email if user else None,
                'authentication_count': user.authentication_count if user else 0
            }
        }
        
        logger.info(f"User authentication result: {success}")
        
        return auth_result
        
    def create_user_with_biometrics(self, username: str, email: Optional[str] = None,
                                  phone: Optional[str] = None,
                                  biometric_data: Dict[str, bytes] = None) -> Dict[str, Any]:
        """
        Create a new user with biometric templates.
        
        Args:
            username: The username of the user
            email: The email address of the user
            phone: The phone number of the user
            biometric_data: A dictionary mapping modalities to biometric data
            
        Returns:
            A dictionary containing the user details
        """
        logger.info(f"Creating new user: {username}")
        
        # Enroll the user in the biometric system
        user_id = self.biometric_authenticator.enroll_user(
            username=username,
            email=email,
            phone=phone
        )
        
        # Add biometric templates if provided
        template_ids = {}
        if biometric_data:
            for modality, data in biometric_data.items():
                template_id = self.biometric_authenticator.add_biometric_template(
                    user_id=user_id,
                    modality=modality,
                    template_data=data
                )
                template_ids[modality] = template_id
                
        # Create a decentralized identity
        identity_id = None
        if biometric_data:
            identity_id = self.decentralized_identity.create_identity(
                biometric_data=biometric_data,
                metadata={
                    'username': username,
                    'email': email,
                    'phone': phone,
                    'user_id': user_id
                }
            )
            
        # Prepare the user creation result
        user_result = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'phone': phone,
            'creation_timestamp': datetime.utcnow().isoformat(),
            'biometric_templates': template_ids,
            'decentralized_identity_id': identity_id
        }
        
        logger.info(f"User created successfully: {user_id}")
        
        return user_result
        
    def train_financial_ai(self, episodes: int = 100, max_steps: int = 1000, verbose: bool = False) -> Dict[str, Any]:
        """
        Train the financial AI agent.
        
        Args:
            episodes: The number of episodes to train for
            max_steps: The maximum number of steps per episode
            verbose: Whether to print training progress
            
        Returns:
            A dictionary containing training metrics
        """
        logger.info(f"Training financial AI agent for {episodes} episodes")
        
        # Train the agent
        training_metrics = self.financial_rl_agent.train(
            env=self.financial_environment,
            episodes=episodes,
            max_steps=max_steps,
            verbose=verbose
        )
        
        # Evaluate the agent
        evaluation_metrics = self.financial_rl_agent.evaluate(
            env=self.financial_environment,
            episodes=10
        )
        
        # Prepare the training result
        training_result = {
            'training_metrics': {
                'rewards': training_metrics['rewards'][-10:],  # Last 10 episodes
                'portfolio_values': training_metrics['portfolio_values'][-10:],  # Last 10 episodes
                'losses': training_metrics['losses'][-10:]  # Last 10 episodes
            },
            'evaluation_metrics': {
                'avg_reward': evaluation_metrics['avg_reward'],
                'avg_portfolio_value': evaluation_metrics['avg_portfolio_value'],
                'avg_return': evaluation_metrics['avg_return'],
                'sharpe_ratio': evaluation_metrics['sharpe_ratio']
            },
            'training_timestamp': datetime.utcnow().isoformat(),
            'episodes': episodes,
            'max_steps': max_steps
        }
        
        logger.info(f"Financial AI agent trained successfully. "
                   f"Avg return: {evaluation_metrics['avg_return']:.4f}, "
                   f"Sharpe ratio: {evaluation_metrics['sharpe_ratio']:.4f}")
        
        return training_result
        
    def detect_transaction_fraud(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential fraud in a transaction using quantum anomaly detection.
        
        Args:
            transaction_data: Transaction data to analyze
            
        Returns:
            A dictionary containing fraud detection results
        """
        logger.info(f"Analyzing transaction for fraud: {transaction_data.get('transaction_id', 'unknown')}")
        
        # Extract features from the transaction
        features = self._extract_transaction_features(transaction_data)
        
        # Detect anomalies
        anomaly_results = self.quantum_anomaly_detection.detect_anomalies(
            features.reshape(1, -1),
            metadata=[{'transaction_id': transaction_data.get('transaction_id', 'unknown')}]
        )
        
        # Prepare the fraud detection result
        fraud_result = {
            'transaction_id': transaction_data.get('transaction_id', 'unknown'),
            'timestamp': datetime.utcnow().isoformat(),
            'is_anomaly': anomaly_results[0]['is_anomaly'],
            'anomaly_score': anomaly_results[0]['score'],
            'threshold': anomaly_results[0]['threshold'],
            'risk_level': 'high' if anomaly_results[0]['score'] > 2.0 else 
                         ('medium' if anomaly_results[0]['score'] > 1.5 else 'low'),
            'recommendation': 'block' if anomaly_results[0]['score'] > 2.0 else 
                             ('review' if anomaly_results[0]['score'] > 1.5 else 'approve')
        }
        
        logger.info(f"Fraud detection result: {fraud_result['recommendation']} "
                   f"(score: {fraud_result['anomaly_score']:.2f})")
        
        return fraud_result
        
    def _extract_transaction_features(self, transaction: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a transaction for anomaly detection.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Feature vector
        """
        # Import numpy for feature extraction
        import numpy as np
        
        # Initialize feature vector (match the feature size of the anomaly detection model)
        features = np.zeros(30)
        
        # Extract basic transaction features
        features[0] = float(transaction.get('amount', 0.0))
        
        # Sender and recipient features (hash-based)
        features[1] = hash(str(transaction.get('sender', ''))) % 1000 / 1000.0
        features[2] = hash(str(transaction.get('recipient', ''))) % 1000 / 1000.0
        
        # Currency feature
        features[3] = hash(str(transaction.get('currency', 'USD'))) % 100 / 100.0
        
        # Timestamp-based features
        if 'timestamp' in transaction:
            try:
                timestamp = datetime.fromisoformat(transaction['timestamp'])
                features[4] = timestamp.hour / 24.0
                features[5] = timestamp.minute / 60.0
                features[6] = timestamp.weekday() / 7.0
                features[7] = timestamp.day / 31.0
                features[8] = timestamp.month / 12.0
            except (ValueError, TypeError):
                # Use current time if timestamp is invalid
                now = datetime.utcnow()
                features[4] = now.hour / 24.0
                features[5] = now.minute / 60.0
                features[6] = now.weekday() / 7.0
                features[7] = now.day / 31.0
                features[8] = now.month / 12.0
        else:
            # Use current time if timestamp is not provided
            now = datetime.utcnow()
            features[4] = now.hour / 24.0
            features[5] = now.minute / 60.0
            features[6] = now.weekday() / 7.0
            features[7] = now.day / 31.0
            features[8] = now.month / 12.0
            
        # Metadata features
        if 'metadata' in transaction and isinstance(transaction['metadata'], dict):
            metadata = transaction['metadata']
            
            # Purpose feature
            if 'purpose' in metadata:
                features[9] = hash(str(metadata['purpose'])) % 100 / 100.0
                
            # Invoice ID feature
            if 'invoice_id' in metadata:
                features[10] = hash(str(metadata['invoice_id'])) % 100 / 100.0
                
            # Other metadata features
            feature_idx = 11
            for key, value in metadata.items():
                if key not in ('purpose', 'invoice_id') and feature_idx < 20:
                    features[feature_idx] = hash(str(key) + str(value)) % 100 / 100.0
                    feature_idx += 1
                    
        return features
        
    def predict_market_trends(self, market_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Predict market trends using quantum neural network.
        
        Args:
            market_data: Historical market data for various assets
            
        Returns:
            A dictionary containing market predictions
        """
        logger.info(f"Predicting market trends for {len(market_data)} assets")
        
        # Import numpy for data processing
        import numpy as np
        
        # Prepare input data for the neural network
        input_features = []
        asset_names = []
        
        for asset_name, prices in market_data.items():
            # Use the last 20 prices (or pad with zeros if fewer)
            asset_prices = np.array(prices[-20:])
            if len(asset_prices) < 20:
                asset_prices = np.pad(asset_prices, (20 - len(asset_prices), 0), 'constant')
                
            # Normalize prices
            if np.max(asset_prices) > 0:
                asset_prices = asset_prices / np.max(asset_prices)
                
            input_features.append(asset_prices)
            asset_names.append(asset_name)
            
        # Convert to numpy array
        input_data = np.array(input_features)
        
        # Make predictions
        predictions = []
        for i in range(len(input_data)):
            prediction = self.quantum_neural_network.predict(input_data[i])
            predictions.append(prediction)
            
        # Interpret predictions
        results = {}
        for i, asset_name in enumerate(asset_names):
            # Interpret the prediction (simplified)
            pred = predictions[i]
            
            # Example interpretation:
            # - First value: short-term trend (1 day)
            # - Second value: medium-term trend (1 week)
            # - Third value: long-term trend (1 month)
            # - Fourth value: volatility prediction
            # - Fifth value: trading volume prediction
            
            short_term = float(pred[0]) if len(pred) > 0 else 0.5
            medium_term = float(pred[1]) if len(pred) > 1 else 0.5
            long_term = float(pred[2]) if len(pred) > 2 else 0.5
            volatility = float(pred[3]) if len(pred) > 3 else 0.5
            volume = float(pred[4]) if len(pred) > 4 else 0.5
            
            results[asset_name] = {
                'short_term_trend': 'up' if short_term > 0.6 else ('down' if short_term < 0.4 else 'neutral'),
                'medium_term_trend': 'up' if medium_term > 0.6 else ('down' if medium_term < 0.4 else 'neutral'),
                'long_term_trend': 'up' if long_term > 0.6 else ('down' if long_term < 0.4 else 'neutral'),
                'predicted_volatility': 'high' if volatility > 0.6 else ('low' if volatility < 0.4 else 'medium'),
                'predicted_volume': 'high' if volume > 0.6 else ('low' if volume < 0.4 else 'medium'),
                'confidence': (short_term + medium_term + long_term) / 3.0
            }
            
        logger.info(f"Market trend predictions generated for {len(results)} assets")
        
        return {
            'predictions': results,
            'timestamp': datetime.utcnow().isoformat(),
            'prediction_horizon': {
                'short_term': '1 day',
                'medium_term': '1 week',
                'long_term': '1 month'
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the status of all system components.
        
        Returns:
            A dictionary containing the status of all system components
        """
        logger.info("Getting system status")
        
        # Get quantum network status
        quantum_network_status = self.quantum_transaction_processor.get_quantum_network_status()
        
        # Prepare the system status
        system_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': (datetime.utcnow() - self.initialization_timestamp).total_seconds(),
            'quantum_computing': {
                'network_status': quantum_network_status['network_health'],
                'active_nodes': quantum_network_status['active_nodes'],
                'qubits_available': quantum_network_status['qubits_available'],
                'qubits_in_use': quantum_network_status.get('qubits_in_use', 0),
                'security_level': self.security_level,
                'quantum_machine_learning': {
                    'neural_network': {
                        'input_size': self.quantum_neural_network.input_size,
                        'hidden_size': self.quantum_neural_network.hidden_size,
                        'output_size': self.quantum_neural_network.output_size,
                        'training_history_count': len(self.quantum_neural_network.training_history)
                    },
                    'anomaly_detection': {
                        'feature_size': self.quantum_anomaly_detection.feature_size,
                        'threshold': float(self.quantum_anomaly_detection.threshold),
                        'anomaly_history_count': len(self.quantum_anomaly_detection.anomaly_history)
                    },
                    'reinforcement_learning': {
                        'state_size': self.quantum_reinforcement_learning.state_size,
                        'action_size': self.quantum_reinforcement_learning.action_size,
                        'epsilon': float(self.quantum_reinforcement_learning.epsilon),
                        'training_history_count': len(self.quantum_reinforcement_learning.training_history)
                    }
                }
            },
            'ai': {
                'financial_environment': {
                    'market_volatility': self.financial_environment.market_volatility,
                    'risk_tolerance': self.financial_environment.risk_tolerance,
                    'transaction_cost': self.financial_environment.transaction_cost
                },
                'financial_agent': {
                    'exploration_rate': self.financial_rl_agent.dqn.exploration_rate,
                    'learning_rate': self.financial_rl_agent.learning_rate,
                    'risk_tolerance': self.financial_rl_agent.risk_tolerance
                }
            },
            'security': {
                'biometric_users': len(self.biometric_authenticator.users),
                'decentralized_identities': len(self.decentralized_identity.identities),
                'security_level': self.security_level
            },
            'transactions': {
                'processed_count': len(self.quantum_transaction_processor.transaction_history)
            }
        }
        
        logger.info("System status retrieved successfully")
        
        return system_status


# Example usage
def example_usage():
    # Create the Pi-Nexus Advanced Integration
    pi_nexus = PiNexusAdvancedIntegration()
    
    # Get the system status
    system_status = pi_nexus.get_system_status()
    print(f"System Status:")
    print(f"  Timestamp: {system_status['timestamp']}")
    print(f"  Uptime: {system_status['uptime']} seconds")
    print(f"  Quantum Computing:")
    print(f"    Network Status: {system_status['quantum_computing']['network_status']}")
    print(f"    Active Nodes: {system_status['quantum_computing']['active_nodes']}")
    print(f"    Qubits Available: {system_status['quantum_computing']['qubits_available']}")
    print(f"    Quantum ML Components:")
    print(f"      Neural Network: {system_status['quantum_computing']['quantum_machine_learning']['neural_network']['input_size']}x"
          f"{system_status['quantum_computing']['quantum_machine_learning']['neural_network']['hidden_size']}x"
          f"{system_status['quantum_computing']['quantum_machine_learning']['neural_network']['output_size']}")
    print(f"      Anomaly Detection: {system_status['quantum_computing']['quantum_machine_learning']['anomaly_detection']['feature_size']} features")
    print(f"      Reinforcement Learning: {system_status['quantum_computing']['quantum_machine_learning']['reinforcement_learning']['state_size']} states, "
          f"{system_status['quantum_computing']['quantum_machine_learning']['reinforcement_learning']['action_size']} actions")
    print(f"  AI:")
    print(f"    Market Volatility: {system_status['ai']['financial_environment']['market_volatility']}")
    print(f"    Risk Tolerance: {system_status['ai']['financial_environment']['risk_tolerance']}")
    print(f"  Security:")
    print(f"    Biometric Users: {system_status['security']['biometric_users']}")
    print(f"    Decentralized Identities: {system_status['security']['decentralized_identities']}")
    
    # Create a user with biometrics
    user_result = pi_nexus.create_user_with_biometrics(
        username="alice",
        email="alice@example.com",
        phone="+1234567890",
        biometric_data={
            "fingerprint": os.urandom(1024),  # Simulated fingerprint data
            "face": os.urandom(2048),  # Simulated face data
            "voice": os.urandom(4096)  # Simulated voice data
        }
    )
    
    print(f"\nUser Created:")
    print(f"  User ID: {user_result['user_id']}")
    print(f"  Username: {user_result['username']}")
    print(f"  Biometric Templates: {len(user_result['biometric_templates'])} templates")
    print(f"  Decentralized Identity ID: {user_result['decentralized_identity_id']}")
    
    # Process a quantum transaction
    transaction = pi_nexus.process_quantum_transaction(
        sender=user_result['user_id'],
        recipient="bob@pinexus.com",
        amount=1000.00,
        currency="USD",
        metadata={
            "purpose": "Invoice payment",
            "invoice_id": "INV-2023-04-15-001"
        }
    )
    
    print(f"\nQuantum Transaction Processed:")
    print(f"  Transaction ID: {transaction['transaction_id']}")
    print(f"  Status: {transaction['status']}")
    print(f"  Settlement Time: {transaction.get('settlement_time', 'N/A')} seconds")
    
    # Detect fraud in the transaction
    fraud_result = pi_nexus.detect_transaction_fraud(transaction)
    
    print(f"\nFraud Detection Results:")
    print(f"  Transaction ID: {fraud_result['transaction_id']}")
    print(f"  Is Anomaly: {fraud_result['is_anomaly']}")
    print(f"  Anomaly Score: {fraud_result['anomaly_score']:.4f}")
    print(f"  Risk Level: {fraud_result['risk_level']}")
    print(f"  Recommendation: {fraud_result['recommendation']}")
    
    # Predict market trends
    market_data = {
        "BTC": [45000, 46000, 47000, 46500, 48000, 49000, 50000, 51000, 52000, 53000,
                54000, 55000, 54500, 54000, 53500, 54000, 55000, 56000, 57000, 58000],
        "ETH": [3000, 3100, 3200, 3150, 3300, 3400, 3500, 3600, 3700, 3800,
                3900, 4000, 3950, 3900, 3850, 3900, 4000, 4100, 4200, 4300],
        "AAPL": [150, 152, 154, 153, 155, 157, 159, 160, 162, 164,
                 166, 168, 167, 166, 165, 166, 168, 170, 172, 174],
        "MSFT": [300, 305, 310, 308, 312, 315, 320, 325, 330, 335,
                 340, 345, 343, 340, 338, 340, 345, 350, 355, 360],
        "AMZN": [3300, 3350, 3400, 3380, 3420, 3450, 3500, 3550, 3600, 3650,
                 3700, 3750, 3730, 3700, 3680, 3700, 3750, 3800, 3850, 3900]
    }
    
    market_predictions = pi_nexus.predict_market_trends(market_data)
    
    print(f"\nMarket Trend Predictions:")
    for asset, prediction in market_predictions['predictions'].items():
        print(f"  {asset}:")
        print(f"    Short-term Trend: {prediction['short_term_trend']}")
        print(f"    Medium-term Trend: {prediction['medium_term_trend']}")
        print(f"    Long-term Trend: {prediction['long_term_trend']}")
        print(f"    Predicted Volatility: {prediction['predicted_volatility']}")
        print(f"    Confidence: {prediction['confidence']:.4f}")
        print()
    
    # Get optimal investment allocation
    allocation = pi_nexus.get_optimal_investment_allocation(user_risk_profile=0.7)
    
    print(f"\nOptimal Investment Allocation:")
    for asset, alloc in sorted(allocation.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {asset}: {alloc:.2%}")
    
    # Train the financial AI (reduced episodes for example)
    training_result = pi_nexus.train_financial_ai(episodes=5, max_steps=100)
    
    print(f"\nFinancial AI Training Results:")
    print(f"  Avg Reward: {training_result['evaluation_metrics']['avg_reward']:.4f}")
    print(f"  Avg Portfolio Value: {training_result['evaluation_metrics']['avg_portfolio_value']:.2f}")
    print(f"  Avg Return: {training_result['evaluation_metrics']['avg_return']:.4f}")
    print(f"  Sharpe Ratio: {training_result['evaluation_metrics']['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    example_usage()