# tests/test_account_service.py
import unittest
from unittest.mock import MagicMock
from services.account_service import AccountService

class TestAccountService(unittest.TestCase):
    def setUp(self):
        self.account_repository = MagicMock()
        self.account_service = AccountService(self.account_repository)

    def test_create_account(self):
        username = "test_user"
        account = self.account_service.create_account(username)
        self.account_repository.save.assert_called_once_with(account)

    def test_get_account(self):
        username = "test_user"
        account = MagicMock()
        self.account_repository.get_by_username.return_value = account
        result = self.account_service.get_account(username)
        self.assertEqual(result, account)
