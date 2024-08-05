import pytest
from web3 import Web3
from energy_trading_contract import EnergyTradingContract
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
    contract = EnergyTradingContract(w3, contract_address)
    return contract

def test_create_trade(contract):
    trade_id = contract.create_trade("0x1234567890abcdef", "0xabcdef1234567890", 100, 200)
    assert trade_id is not None

def test_get_trade(contract):
    trade_id = 1
    trade = contract.get_trade(trade_id)
    assert trade is not None
    assert trade["seller"] == "0x1234567890abcdef"
    assert trade["buyer"] == "0xabcdef1234567890"
    assert trade["energy_amount"] == 100
    assert trade["price"] == 200

def test_execute_trade(contract):
    trade_id = 1
    contract.execute_trade(trade_id)
    trade = contract.get_trade(trade_id)
    assert trade["status"] == "EXECUTED"

def test_cancel_trade(contract):
    trade_id = 1
    contract.cancel_trade(trade_id)
    trade = contract.get_trade(trade_id)
    assert trade["status"] == "CANCELED"

def test_get_all_trades(contract):
    trades = contract.get_all_trades()
    assert len(trades) >= 1
