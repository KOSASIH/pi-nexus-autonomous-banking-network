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

# Load contract ABI and bytecode
with open('contracts/PiTradeToken.sol/PiTradeToken.abi', 'r') as f:
    abi = json.load(f)
with open('contracts/PiTradeToken.sol/PiTradeToken.bin', 'r') as f:
    bytecode = f.read()

# Deploy contract
account = Account.from_key(PRIVATE_KEY)
tx_hash = w3.eth.contract(abi=abi, bytecode=bytecode).constructor().transact({'from': account.address, 'gas': 2000000})
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# Print contract address
print(f'Contract deployed to {tx_receipt.contractAddress}')
