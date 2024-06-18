from stellar_smart_contract import StellarSmartContract
import requests

class PiNetworkOracle:
    def __init__(self, contract_seed: str, network: str, pi_network_api_url: str):
        self.stellar_contract = StellarSmartContract(contract_seed, network)
        self.pi_network_api_url = pi_network_api_url

    def get_pi_network_balance(self, user_id: str):
        """Get the Pi Network balance for a user"""
        response = requests.get(f"{self.pi_network_api_url}/users/{user_id}/balance")
        return response.json()["balance"]

    def execute_pi_network_transaction(self, source_user_id: str, destination_user_id: str, amount: int):
        """Execute a Pi Network transaction using the Stellar smart contract"""
        source_pi_balance = self.get_pi_network_balance(source_user_id)
        if source_pi_balance >= amount:
            destination_pi_address = self.get_pi_network_address(destination_user_id)
            self.stellar_contract.execute_payment(
                source_user_id, destination_pi_address, amount, "PI"
            )
            return True
        return False

    def get_pi_network_address(self, user_id: str):
        """Get the Pi Network address for a user"""
        response = requests.get(f"{self.pi_network_api_url}/users/{user_id}/address")
        return response.json()["address"]
