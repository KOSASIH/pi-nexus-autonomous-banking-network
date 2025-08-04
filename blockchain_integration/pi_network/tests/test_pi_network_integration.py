# tests/test_pi_network_integration.py

import unittest
from unittest.mock import MagicMock, patch

from pi_network_integration import PiNetworkAPI


class TestPiNetworkIntegration(unittest.TestCase):
    def setUp(self):
        self.api = PiNetworkAPI("https://api.minepi.com/v2", "your_api_key_here")

    def test_get_balance(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"balance": 100}
            balance = self.api.get_balance("your_address_here")
            self.assertEqual(balance, 100)

    def test_submit_transaction(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"tx_id": "your_tx_id_here"}
            tx_id = self.api.submit_transaction(
                "from_address_here", "to_address_here", 100
            )
            self.assertEqual(tx_id, "your_tx_id_here")

    def test_get_transaction(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "tx_id": "your_tx_id_here",
                "status": "pending",
            }
            tx_info = self.api.get_transaction("your_tx_id_here")
            self.assertEqual(tx_info["status"], "pending")


if __name__ == "__main__":
    unittest.main()
