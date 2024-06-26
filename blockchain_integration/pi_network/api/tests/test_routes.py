import unittest
from pi_network.api.routes import api
from pi_network.api.models import User, Account, Transaction

class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.client = api.test_client()

    def test_user_routes(self):
        response = self.client.post('/users', json={'username': 'test_user', 'password': 'test_password'})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {'id': 1, 'username': 'test_user'})

        response = self.client.get('/users/1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'id': 1, 'username': 'test_user'})

    def test_account_routes(self):
        response = self.client.post('/accounts', json={'user_id': 1, 'balance': 100.0})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {'id': 1, 'user_id': 1, 'balance': 100.0})

        response = self.client.get('/accounts/1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'id': 1, 'user_id': 1, 'balance': 100.0})

    def test_transaction_routes(self):
        response = self.client.post('/transactions', json={'account_id': 1, 'amount': 50.0})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {'id': 1, 'account_id': 1, 'amount': 50.0})

        response = self.client.get('/transactions/1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'id': 1, 'account_id': 1, 'amount': 50.0})

if __name__ == '__main__':
    unittest.main()
