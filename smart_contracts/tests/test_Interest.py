import pytest
from brownie import Interest

def test_calculate_interest():
    interest = Interest.deploy({'from': accounts[0]})
    assert interest.calculateInterest(100, 10) == 110
    assert interest.calculateInterest(100, 20) == 120
    assert interest.calculateInterest(100, 30) == 130

def test_only_owner():
    interest = Interest.deploy({'from': accounts[0]})
    with pytest.raises(Exception):
        interest.setInterestRate(20, {'from': accounts[1]})
