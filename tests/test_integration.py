# tests/test_integration.py
import unittest
from unittest.mock import patch
from transaction_processing.tasks import process_transaction
from api_gateway.app import create_app

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    @patch('transaction_processing.tasks.process_transaction')
    def test_process_transaction(self, mock_process_transaction):
        # ...
