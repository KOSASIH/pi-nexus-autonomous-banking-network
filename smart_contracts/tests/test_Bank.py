import pytest
from brownie import Bank

def test_deposit():
    bank = Bank.deploy({'from': accounts[0]})
    initial_balance = accounts[1].balance()
    bank.deposit({'from': accounts[1], 'value': 100})
    final_balance = accounts[1].balance()
    assert final_balance == initial_balance - 100
    assert bank.balances(accounts[1]) == 100

def test_withdraw():
    bank = Bank.deploy({'from': accounts[0]})
    bank.deposit({'from': accounts[1], 'value': 100})
    initial_balance = bank.balances(accounts[1])
    bank.withdraw(100, {'from': accounts[1]})
    final_balance = bank.balances(accounts[1])
    assert final_balance == 0

def test_only_owner():
    bank = Bank.deploy({'from': accounts[0]})
    with pytest.raises(Exception):
        bank.withdraw(100, {'from': accounts[1]})
