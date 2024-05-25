import os
import json
from web3 import Web3
from eth_sig_util import to_checksum_address
from cryptography.hazmat.primitives import serialization

def secure_send_transaction(web3, private_key, to_address, value):
    # Securely send a transaction using Web3.py
    tx_count = web3.eth.get_transaction_count(to_checksum_address(private_key))
    tx = {
        'from': to_checksum_address(private_key),
        'to': to_address,
        'value': value,
        'gas': 20000,
        'gasPrice': web3.eth.gas_price,
        'nonce': tx_count
    }
    signed_tx = web3.eth.account.sign_transaction(tx, private_key)
    web3.eth.send_raw_transaction(signed_tx.rawTransaction)

def secure_generate_keypair():
    # Securely generate a keypair using cryptography
    private_key = serialization.generate_private_key(
        algorithm=serialization.RSA(),
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key
