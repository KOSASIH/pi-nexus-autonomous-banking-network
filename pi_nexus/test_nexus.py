import pytest
from nexus import Nexus
from config import API_ENDPOINT

@pytest.fixture
def nexus():
    nexus = Nexus()
    nexus.init_app({
        "TESTING": True,
        "API_ENDPOINT": API_ENDPOINT
    })
    nexus.create_tables()
    yield nexus
    nexus.drop_tables()

def test_get_account_balance(nexus):
    account_id = "12345"
    balance = 100.0
    nexus.add_account(account_id, balance)
    assert nexus.get_account_balance(account_id) == balance

def test_get_account_transactions(nexus):
    account_id = "12345"
    transactions = ["Transaction 1", "Transaction 2"]
    nexus.add_account_transactions(account_id, transactions)
    assert nexus.get_account_transactions(account_id) == transactions
