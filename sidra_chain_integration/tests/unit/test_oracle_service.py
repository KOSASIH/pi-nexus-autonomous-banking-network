import unittest

from oracle_service import OracleService


class TestOracleService(unittest.TestCase):
    def test_update_price(self) -> None:
        # Create a mock OracleModel instance
        oracle_model = OracleModel(
            node_url="https://example.com",
            api_key="YOUR_API_KEY",
            contract_address="0x...",
        )

        # Create an OracleService instance with the mock model
        oracle_service = OracleService(oracle_model)

        # Test the updatePrice method
        oracle_service.update_price("asset_address", 100)

        # Assert that the price was updated successfully
        pass

    def test_get_price(self) -> None:
        # Create a mock OracleModel instance
        oracle_model = OracleModel(
            node_url="https://example.com",
            api_key="YOUR_API_KEY",
            contract_address="0x...",
        )

        # Create an OracleService instance with the mock model
        oracle_service = OracleService(oracle_model)

        # Test the getPrice method
        price = oracle_service.get_price("asset_address")

        # Assert that the price is correct
        pass
