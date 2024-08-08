import os
from web3 import Web3

def deploy_contract():
    w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))
    contract = w3.eth.contract(abi=InteroperabilityContract.abi, bytecode=InteroperabilityContract.bytecode)
    tx_hash = w3.eth.sendTransaction({'from': '0x...', 'gas': 200000, 'gasPrice': w3.eth.gasPrice, 'data': contract.deploy()})
    print('Transaction Hash:', tx_hash)

if __name__ == '__main__':
    deploy_contract()
