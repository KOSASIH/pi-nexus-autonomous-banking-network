import os
import json
from web3 import Web3, HTTPProvider

def get_web3_provider():
    return Web3(HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))

def get_contract_instance(contract_address, abi):
    w3 = get_web3_provider()
    return w3.eth.contract(address=contract_address, abi=abi)

def deploy_contract(w3, contract_interface):
    tx_hash = w3.eth.contract(bytecode=contract_interface['bytecode'], abi=contract_interface['abi']).constructor().transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt.contractAddress

def load_contract_interface(contract_name):
    with open(f'contracts/{contract_name}.json', 'r') as f:
        return json.load(f)

def get_contract_address(contract_name):
    with open(f'contracts/{contract_name}_address.txt', 'r') as f:
        return f.read()

def save_contract_address(contract_name, contract_address):
    with open(f'contracts/{contract_name}_address.txt', 'w') as f:
        f.write(contract_address)
