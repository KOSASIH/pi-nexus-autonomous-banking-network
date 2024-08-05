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

# Migrate contract state
tx_hash = contract.functions.migrate().transact({'from': Account.from_key(PRIVATE_KEY).address, 'gas': 2000000})
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# Print migration result
print(f'Migration successful: {tx_receipt.status}')
