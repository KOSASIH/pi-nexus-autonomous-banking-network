import os
from web3 import Web3

def get_token_balance(web3, token_address, address):
    token_contract = web3.eth.contract(address=token_address, abi=ERC20.abi)
    return token_contract.functions.balanceOf(address).call()

def transfer_token(web3, token_address, from_address, to_address, value):
    token_contract = web3.eth.contract(address=token_address, abi=ERC20.abi)
    tx_hash = token_contract.functions.transfer(to_address, value).transact({'from': from_address})
    return tx_hash
