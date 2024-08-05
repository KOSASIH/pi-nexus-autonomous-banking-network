import os
import json
from web3 import Web3
from web3.auto import w3
from eth_account import Account

# Load configuration from environment variables
NETWORK = os.environ['NETWORK']
PRIVATE_KEY = os.environ['PRIVATE_KEY']
CONTRACT_ADDRESS = os.environ['CONTRACT_ADDRESS']

# Set up Web3 provider
w3 = Web3(Web3.HTTPProvider(f'https://{NETWORK}.infura.io/v3/{os.environ["INFURA_PROJECT_ID"]}'))

# Load contract ABI
with open('contracts/PiTradeToken.sol/PiTradeToken.abi', 'r') as f:
    abi = json.load(f)

# Create contract instance
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

# Run tests
def test_token_name():
    assert contract.functions.name().call() == 'PiTrade Token'

def test_token_symbol():
    assert contract.functions.symbol().call() == 'PTT'

def test_token_total_supply():
    assert contract.functions.totalSupply().call() == 1000000

def test_mint_tokens():
    account = Account.from_key(PRIVATE_KEY)
    tx_hash = contract.functions.mint(account.address, 100).transact({'from': account.address, 'gas': 2000000})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert contract.functions.balanceOf(account.address).call() == 100

def test_transfer_tokens():
    account1 = Account.from_key(PRIVATE_KEY)
    account2 = Account.from_key(os.environ['PRIVATE_KEY_2'])
    tx_hash = contract.functions.transfer(account2.address, 50).transact({'from': account1.address, 'gas': 2000000})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert contract.functions.balanceOf(account1.address).call() == 50
    assert contract.functions.balanceOf(account2.address).call() == 50

# Run all tests
test_token_name()
test_token_symbol()
test_token_total_supply()
test_mint_tokens()
test_transfer_tokens()

print('All tests passed!')
