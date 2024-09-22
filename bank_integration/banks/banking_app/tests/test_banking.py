import unittest
from app.banking import get_account_balance, transfer_funds

class TestBanking(unittest.TestCase):
    def test_get_account_balance(self):
        # Test getting an account balance
        account_id = 1
        balance = get_account_balance(account_id)
        self.assertIsNotNone(balance)

    def test_transfer_funds(self):
        # Test transferring funds between accounts
        sender_account_id = 1
        recipient_account_id = 2
        amount = 100.0
        transfer_funds(sender_account_id, recipient_account_id, amount)
        # Assert that the funds were transferred successfully

if __name__ == "__main__":
    unittest.main()
