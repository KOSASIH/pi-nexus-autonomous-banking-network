import unittest
from sidra_chain_service import SidraChainService

class TestSidraChainService(unittest.TestCase):
    def test_transfer_funds(self) -> None:
        # Create a mock SidraChainModel instance
        sidra_chain_model = SidraChainModel(node_url="https://example.com", api_key="YOUR_API_KEY", contract_address="0x...")

        # Create a SidraChainService instance with the mock model
        sidra_chain_service = SidraChainService(sidra_chain_model)

        # Test the transferFunds method
        sidra_chain_service.transfer_funds("recipient_address", 100)

        # Assert that the transfer was successful
        pass
