import os
import sys
from pi_network.pi_network_core import PiNetworkCore
from pi_network.pi_network_api import PiNetworkAPI
from pi_network.pi_network_db import PiNetworkDB
from pi_network.config.environment_variables import EnvironmentVariables
from pi_network.config.network_settings import NetworkSettings
from pi_network.utils.cryptographic_helpers import generate_key_pair
from pi_network.utils.data_processing_helpers import load_data, preprocess_data

def main():
    # Load environment variables and network settings
    env_vars = EnvironmentVariables()
    network_settings = NetworkSettings()

    # Generate key pair for encryption and decryption
    private_key, public_key = generate_key_pair()

    # Initialize PiNetworkCore, PiNetworkAPI, and PiNetworkDB
    pi_network_core = PiNetworkCore(env_vars, network_settings)
    pi_network_api = PiNetworkAPI(pi_network_core, public_key)
    pi_network_db = PiNetworkDB(env_vars, network_settings)

    # Load and preprocess data
    data = load_data('data.csv')
    data = preprocess_data(data)

    # Start the Pi-Nexus Autonomous Banking Network
    pi_network_core.start()
    pi_network_api.start()
    pi_network_db.start()

    # Run the network
    while True:
        # Process transactions and update the database
        pi_network_core.process_transactions()
        pi_network_db.update_database()

        # Sleep for 1 second
        time.sleep(1)

if __name__ == '__main__':
    main()
