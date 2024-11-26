import unittest
from src.smart_contracts.transaction_contract import TransactionContract

class TestSmartContracts(unittest.TestCase):
    def setUp(self):
        self.contract = TransactionContract()

    def test_execute_transaction(self):
        result = self.contract.execute_transaction("address1", "address2", 10)
        self.assertTrue(result)

    def test_get_balance(self):
        balance = self.contract.get_balance("address1")
        self.assertEqual(balance, 100)  # Assuming initial balance is 100

if __name__ == '__main__':
    unittest.main()
