import unittest

from app import create_app

from api.endpoints import endpoints
from api.serializers import TransactionSchema, UserSchema


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client.testing = True

    def tearDown(self):
        self.app_context.pop()

    def test_create_user(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        response = self.client.post("/api/users", json=user_data)
        self.assertEqual(response.status_code, 201)
        user_schema = UserSchema()
        user = user_schema.load(response.get_json())
        self.assertIsNotNone(user.id)

    def test_get_user(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        self.client.post("/api/users", json=user_data)
        response = self.client.get("/api/users/1")
        self.assertEqual(response.status_code, 200)
        user_schema = UserSchema()
        user = user_schema.load(response.get_json())
        self.assertEqual(user.name, "Test User")
        self.assertEqual(user.email, "test@example.com")

    def test_update_user(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        self.client.post("/api/users", json=user_data)
        user_data["name"] = "Updated Test User"
        response = self.client.put("/api/users/1", json=user_data)
        self.assertEqual(response.status_code, 200)
        user_schema = UserSchema()
        user = user_schema.load(response.get_json())
        self.assertEqual(user.name, "Updated Test User")

    def test_delete_user(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        self.client.post("/api/users", json=user_data)
        response = self.client.delete("/api/users/1")
        self.assertEqual(response.status_code, 204)

    def test_create_transaction(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        self.client.post("/api/users", json=user_data)
        transaction_data = {"user_id": 1, "amount": 100.0}
        response = self.client.post("/api/transactions", json=transaction_data)
        self.assertEqual(response.status_code, 201)
        transaction_schema = TransactionSchema()
        transaction = transaction_schema.load(response.get_json())
        self.assertIsNotNone(transaction.id)

    def test_get_transactions(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        self.client.post("/api/users", json=user_data)
        transaction_data = {"user_id": 1, "amount": 100.0}
        self.client.post("/api/transactions", json=transaction_data)
        response = self.client.get("/api/transactions")
        self.assertEqual(response.status_code, 200)
        transaction_schema = TransactionSchema(many=True)
        transactions = transaction_schema.load(response.get_json())
        self.assertEqual(len(transactions), 1)

    def test_get_transaction(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        self.client.post("/api/users", json=user_data)
        transaction_data = {"user_id": 1, "amount": 100.0}
        self.client.post("/api/transactions", json=transaction_data)
        response = self.client.get("/api/transactions/1")
        self.assertEqual(response.status_code, 200)
        transaction_schema = TransactionSchema()
        transaction = transaction_schema.load(response.get_json())
        self.assertEqual(transaction.amount, 100.0)

    def test_update_transaction(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        self.client.post("/api/users", json=user_data)
        transaction_data = {"user_id": 1, "amount": 100.0}
        self.client.post("/api/transactions", json=transaction_data)
        transaction_data["amount"] = 200.0
        response = self.client.put("/api/transactions/1", json=transaction_data)
        self.assertEqual(response.status_code, 200)
        transaction_schema = TransactionSchema()
        transaction = transaction_schema.load(response.get_json())
        self.assertEqual(transaction.amount, 200.0)

    def test_delete_transaction(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword",
        }
        self.client.post("/api/users", json=user_data)
        transaction_data = {"user_id": 1, "amount": 100.0}
        self.client.post("/api/transactions", json=transaction_data)
        response = self.client.delete("/api/transactions/1")
        self.assertEqual(response.status_code, 204)
