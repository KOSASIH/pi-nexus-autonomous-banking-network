# tests/test_wallet.py
import unittest

from wallet import Wallet


class TestWallet(unittest.TestCase):
    def test_add_coins(self):
        wallet = Wallet()
        wallet.add_coins(10)
        self.assertEqual(wallet.get_balance(), 10)

    def test_get_balance(self):
        wallet = Wallet()
        wallet.add_coins(10)
        self.assertEqual(wallet.get_balance(), 10)


if __name__ == "__main__":
    unittest.main()
