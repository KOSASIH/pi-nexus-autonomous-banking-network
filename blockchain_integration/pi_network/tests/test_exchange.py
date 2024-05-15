import unittest
from unittest.mock import MagicMock, patch

from exchange import Exchange


class TestExchange(unittest.TestCase):
    def setUp(self):
        self.exchange = Exchange()

    def test_get_rate_success(self):
        with patch("exchange.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"rate": 1.0}
            rate = self.exchange.get_rate("USD")
            self.assertEqual(rate, 1.0)
            mock_get.assert_called_once_with(
                "https://api.exchangerate-api.com/v4/latest/USD"
            )

    def test_get_rate_invalid_currency(self):
        with self.assertRaises(ValueError):
            self.exchange.get_rate("Invalid Currency")

    def test_convert_amount_success(self):
        with patch("exchange.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"rate": 1.0}
            amount = self.exchange.convert_amount(100, "USD", "EUR")
            self.assertEqual(amount, 100.0)

    def test_convert_amount_invalid_currency(self):
        with self.assertRaises(ValueError):
            self.exchange.convert_amount(100, "USD", "Invalid Currency")
