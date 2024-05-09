import pytest
from brownie import ContractA

def test_set_and_get():
    contract = ContractA.deploy()
    contract.set(10)
    assert contract.get() == 10

def test_set_and_get_different_values():
    contract = ContractA.deploy()
    contract.set(10)
    contract.set(20)
    assert contract.get() == 20
