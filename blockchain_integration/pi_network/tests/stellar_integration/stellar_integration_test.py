import unittest
from stellar_integration.stellar_integration import StellarIntegration

class TestStellarIntegration(unittest.TestCase):
    def test_get_account_balance(self):
        si = StellarIntegration()
        account_id = "GA...123"
        balance = si.get_account_balance(account_id)
        self.assertIsNotNone(balance)

    def test_send_transaction(self):
        si = StellarIntegration()
        source_account_id = "GA...123"
        dest_account_id = "GA...456"
        amount = 10.0
        memo = "Test transaction"
        tx_hash = si.send_transaction(source_account_id, dest_account_id, amount, memo)
        self.assertIsNotNone(tx_hash)

    def test_get_transaction_history(self):
        si = StellarIntegration()
        account_id = "GA...123"
        tx_history = si.get_transaction_history(account_id)
        self.assertIsNotNone(tx_history)

if __name__ == '__main__':
    unittest.main()
