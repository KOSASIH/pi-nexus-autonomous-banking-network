import unittest
from defi import DeFiManager  # Assuming you have a DeFiManager class

class TestDeFiManager(unittest.TestCase):
    def setUp(self):
        self.defi_manager = DeFiManager()

    def test_stake_tokens(self):
        result = self.defi_manager.stake_tokens("Alice", 100)
        self.assertTrue(result)

    def test_withdraw_tokens(self):
        self.defi_manager.stake_tokens("Bob", 200)
        result = self.defi_manager.withdraw_tokens("Bob", 100)
        self.assertTrue(result)

    def test_insufficient_balance(self):
        with self.assertRaises(ValueError):
            self.defi_manager.withdraw_tokens("Charlie", 100)

if __name__ == "__main__":
    unittest.main()
