import time
import requests
from typing import Dict, Any

class Explorer:
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize the Explorer class with a dictionary of API keys.
        """
        self.api_keys = api_keys

    def get_blockchain_info(self) -> Dict[str, Any]:
        """
        Get information about the Pi-Nexus Autonomous Banking Network blockchain.
        """
        url = "http://localhost:8080/api/blockchain-info"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['pi-nexus']}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_block_by_number(self, block_number: int) -> Dict[str, Any]:
        """
        Get information about a block in the Pi-Nexus Autonomous Banking Network blockchain.
        """
        url = f"http://localhost:8080/api/block-by-number/{block_number}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['pi-nexus']}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_transaction_by_id(self, transaction_id: str) -> Dict[str, Any]:
        """
        Get information about a transaction in the Pi-Nexus Autonomous Banking Network blockchain.
        """
        url = f"http://localhost:8080/api/transaction-by-id/{transaction_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['pi-nexus']}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_smart_contract_by_address(self, contract_address: str) -> Dict[str, Any]:
        """
        Get information about a smart contract in the Pi-Nexus Autonomous Banking Network blockchain.
        """
        url = f"http://localhost:8080/api/smart-contract-by-address/{contract_address}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['pi-nexus']}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

# Example usage
api_keys = {
    "pi-nexus": "your_pi_nexus_api_key"
}
explorer = Explorer(api_keys)

# Get information about the Pi-Nexus Autonomous Banking Network blockchain
blockchain_info = explorer.get_blockchain_info()
print(blockchain_info)

# Get information about a block in the Pi-Nexus Autonomous Banking Network blockchain
block_info = explorer.get_block_by_number(12345)
print(block_info)

# Get information about a transaction in the Pi-Nexus Autonomous Banking Network blockchain
transaction_info = explorer.get_transaction_by_id("your_transaction_id")
print(transaction_info)

# Get information about a smart contract in the Pi-Nexus Autonomous Banking Network blockchain
smart_contract_info = explorer.get_smart_contract_by_address("your_contract_address")
print(smart_contract_info)
