import pytest
from brownie import ContractA, ContractC

def test_set_and_get_contract_a_data():
    contract_a = ContractA.deploy()
    contract_c = ContractC.deploy(contract_a.address)
    contract_c.setContractAData(10)
    assert contract_a.get() == 10

def test_set_and_get_contract_a_data_different_values():
    contract_a = ContractA.deploy()
    contract_c = ContractC.deploy(contract_a.address)
    contract_c.setContractAData(10)
    contract_c.setContractAData(20)
    assert contract_a.get() == 20
