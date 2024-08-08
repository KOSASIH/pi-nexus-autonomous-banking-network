import os
from web3 import Web3

def get_web3_provider(network):
    if network == 'ethereum':
        return Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID')
    elif network == 'binance_smart_chain':
        return Web3.HTTPProvider('https://bsc-dataseed.binance.org/api/v1/')
    else:
        raise ValueError('Unsupported network')

def get_contract_address(network, contract_name):
    if network == 'ethereum' and contract_name == 'InteroperabilityContract':
        return '0x...'
    elif network == 'binance_smart_chain' and contract_name == 'InteroperabilityContract':
        return '0x...'
    else:
        raise ValueError('Unsupported contract')

def get_token_address(network, token_name):
    if network == 'ethereum' and token_name == 'ERC20Token':
        return '0x...'
    elif network == 'binance_smart_chain' and token_name == 'BEP20Token':
        return '0x...'
    else:
        raise ValueError('Unsupported token')
