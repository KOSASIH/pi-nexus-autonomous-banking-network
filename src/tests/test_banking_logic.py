import unittest

from banking_logic import BankAccount


class TestBankingLogic(unittest.TestCase):
    def setUp(self):
        self.account = BankAccount()

    def test_initial_balance(self):
        self.assertEqual(self.account.balance, 0)

    def test_deposit(self):
        self.account.deposit(1000)
        self.assertEqual(self.account.balance, 1000)

    def test_withdraw(self):
        self.account.deposit(1000)
        self.account.withdraw(500)
        self.assertEqual(self.account.balance, 500)

    def test_insufficient_balance(self):
        with self.assertRaises(ValueError):
            self.account.withdraw(1000)

    def test_display_balance(self):
        self.account.deposit(1000)
        self.account.display()
        # Verify that the display method prints the correct balance


if __name__ == "__main__":
    unittest.main()
