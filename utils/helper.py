import os
import json
from web3 import Web3
from web3.middleware import geth_poa_middleware

def get_contract_abi(contract_name):
    with open(f'./build/contracts/{contract_name}.abi', 'r') as f:
        return json.load(f)

def get_contract_address(contract_name):
    with open(f'./build/contracts/{contract_name}.address', 'r') as f:
        return f.read().strip()

def get_web3_provider():
    infura_project_id = os.environ.get('INFURA_PROJECT_ID', '')
    web3_provider = Web3(Web3.HTTPProvider(f'https://rinkeby.infura.io/v3/{infura_project_id}'))
    web3_provider.middleware_onion.inject(geth_poa_middleware, layer=0)
    return web3_provider

def get_contract_instance(contract_name):
    abi = get_contract_abi(contract_name)
    address = get_contract_address(contract_name)
    web3_provider = get_web3_provider()
    contract = web3_provider.eth.contract(address=address, abi=abi)
    return contract
