import unittest
from fnex.models import User, Account, Transaction

class TestFineXModels(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.user = User.objects.create(username='testuser', email='testuser@example.com')
        self.account = Account.objects.create(user=self.user, account_number='123456789', balance=1000)
        self.transaction = Transaction.objects.create(account=self.account, amount=-50, description='Test transaction')

    def test_user_model(self):
        # Test User model
        self.assertEqual(self.user.username, 'testuser')
        self.assertEqual(self.user.email, 'testuser@example.com')

    def test_account_model(self):
        # Test Account model
        self.assertEqual(self.account.user, self.user)
        self.assertEqual(self.account.account_number, '123456789')
        self.assertEqual(self.account.balance, 1000)

    def test_transaction_model(self):
        # Test Transaction model
        self.assertEqual(self.transaction.account, self.account)
        self.assertEqual(self.transaction.amount, -50)
        self.assertEqual(self.transaction.description, 'Test transaction')

if __name__ == '__main__':
    unittest.main()
