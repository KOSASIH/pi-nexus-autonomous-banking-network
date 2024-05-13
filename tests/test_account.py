# tests/test_account.py
import pytest
from models.account import Account

def test_account_deposit():
    account = Account("1234567890", 1000)
    account.deposit(500)
    assert account.balance == 1500

def test_account_withdraw():
    account = Account("1234567890", 1000)
    account.withdraw(500)
    assert account.balance == 500
