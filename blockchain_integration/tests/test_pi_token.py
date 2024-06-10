import pytest
from web3 import Web3
from blockchain_integration.contracts.optimized_pi_token import OptimizedPiToken

@pytest.fixture
def w3():
    return Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))

@pytest.fixture
def pi_token(w3):
    pi_token_contract = OptimizedPiToken(initialSupply=1000000, constructor_args=[w3.eth.accounts[0]])
    return pi_token_contract

def test_total_supply(w3, pi_token):
    assert pi_token.totalSupply() == 1000000

def test_balance_of(w3, pi_token):
    assert pi_token.balanceOf(w3.eth.accounts[0]) == 1000000

def test_transfer(w3, pi_token):
    pi_token.transfer(w3.eth.accounts[1], 10000)
    assert pi_token.balanceOf(w3.eth.accounts[0]) == 990000
    assert pi_token.balanceOf(w3.eth.accounts[1]) == 10000

def test_approve_and_transfer_from(w3, pi_token):
    pi_token.approve(w3.eth.accounts[1], 10000)
    pi_token.transferFrom(w3.eth.accounts[0], w3.eth.accounts[1], 10000)
    assert pi_token.balanceOf(w3.eth.accounts[0]) == 990000
    assert pi_token.balanceOf(w3.eth.accounts[1]) == 20000
