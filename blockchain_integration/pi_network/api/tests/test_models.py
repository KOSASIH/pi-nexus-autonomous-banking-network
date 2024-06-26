import unittest
from pi_network.api.models import User, Account, Transaction

class TestModels(unittest.TestCase):
    def setUp(self):
        self.user = User(username='test_user', password='test_password')
        self.account = Account(user=self.user, balance=100.0)
        self.transaction = Transaction(account=self.account, amount=50.0)

    def test_user_model(self):
        self.assertEqual(self.user.username, 'test_user')
        self.assertEqual(self.user.password, 'test_password')

    def test_account_model(self):
        self.assertEqual(self.account.user, self.user)
        self.assertEqual(self.account.balance, 100.0)

    def test_transaction_model(self):
        self.assertEqual(self.transaction.account, self.account)
        self.assertEqual(self.transaction.amount, 50.0)

    def test_model_relationships(self):
        self.assertEqual(self.account.user, self.user)
        self.assertEqual(self.transaction.account, self.account)

if __name__ == '__main__':
    unittest.main()
