import pytest
from brownie import ContractB

def test_set_and_get():
    contract = ContractB.deploy()
    contract.set(10)
    assert contract.get() == 10

def test_set_and_get_different_values():
    contract = ContractB.deploy()
    contract.set(10)
    contract.set(20)
    assert contract.get() == 20

def test_inherited_contract_a_functions():
    contract = ContractB.deploy()
    contract_a = ContractA.at(contract.address)
    contract_a.set(10)
    assert contract_a.get() == 10
    assert contract.get() == 10
