# Main entry point for PiHE
import os
from config.environment import PI_NETWORK_RPC_URL, PI_NETWORK_CHAIN_ID
from crypto.key_management import KeyManager
from models.homomorphic_model import HomomorphicModel
from models.data_processing import DataProcessor
from applications.defi.defi_app import DeFiApp
from applications.healthcare.healthcare_app import HealthcareApp

def main():
    # Initialize key manager and homomorphic model for PiHE
    key_manager = KeyManager(2048, 4096)
    homomorphic_model = HomomorphicModel(key_manager.he)

    # Initialize data processor and applications for PiHE
    data_processor = DataProcessor(homomorphic_model)
    defi_app = DeFiApp(data_processor)
    healthcare_app = HealthcareApp(data_processor)

    # Load encrypted data from Pi Network
    encrypted_data = load_encrypted_data_from_pi_network(PI_NETWORK_RPC_URL, PI_NETWORK_CHAIN_ID)

    # Process encrypted data using homomorphic model
    processed_data = data_processor.process_data(encrypted_data)

    # Execute DeFi application
    defi_app.execute_trade(processed_data)

    # Execute Healthcare application
    healthcare_app.analyze_patient_data(processed_data)

if __name__ == '__main__':
    main()
