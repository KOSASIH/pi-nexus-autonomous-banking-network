import unittest
from scripts.coin_minting import mint_coin

class TestCoinMinting(unittest.TestCase):
    def test_mint_coin(self):
        coin = mint_coin("Test Coin", "TST", 10.0)
        self.assertIsInstance(coin, Coin)
        self.assertEqual(coin.name, "Test Coin")
        self.assertEqual(coin.symbol, "TST")
        self.assertEqual(coin.amount, 10.0)
