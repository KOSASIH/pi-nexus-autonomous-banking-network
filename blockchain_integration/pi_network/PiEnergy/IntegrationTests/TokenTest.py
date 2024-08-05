import pytest
from web3 import Web3
from token_contract import TokenContract
from web3.middleware import geth_poa_middleware

def pytest_addoption(parser):
    parser.addoption("--contract-address", action="store", default="0x742d35Cc6634C0532925a3b844Bc454e4438f44e")
    parser.addoption("--infura-project-id", action="store", default="YOUR_PROJECT_ID")

@pytest.fixture
def w3(request):
    w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{request.config.getoption('--infura-project-id')}"))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3

@pytest.fixture
def contract(w3):
    contract_address = w3.toChecksumAddress(w3.config["contract_address"])
    contract = TokenContract(w3, contract_address)
    return contract

def test_name(contract):
    name = contract.name()
    assert name == "EnergyToken"

def test_symbol(contract):
    symbol = contract.symbol()
    assert symbol == "ETK"

def test_total_supply(contract):
    total_supply = contract.total_supply()
    assert total_supply >= 1000000

def test_balance_of(contract):
    address = "0x1234567890abcdef"
    balance = contract.balance_of(address)
    assert balance >= 100

def test_transfer(contract):
    sender = "0x1234567890abcdef"
    recipient = "0xabcdef1234567890"
    amount = 100
    contract.transfer(sender, recipient, amount)
    balance_sender = contract.balance_of(sender)
    balance_recipient = contract.balance_of(recipient)
    assert balance_sender == 0
    assert balance_recipient == amount

def test_approve(contract):
    owner = "0x1234567890abcdef"
    spender = "0xabcdef1234567890"
    amount = 100
    contract.approve(owner, spender, amount)
    allowance = contract.allowance(owner, spender)
    assert allowance == amount

def test_transfer_from(contract):
    sender = "0x1234567890abcdef"
    recipient = "0xabcdef1234567890"
    amount = 100
    contract.transfer_from(sender, recipient, amount)
    balance_sender = contract.balance_of(sender)
    balance_recipient = contract.balance_of(recipient)
    assert balance_sender == 0
    assert balance_recipient == amount
