import pytest
from web3 import Web3
from blockchain.ethereum import Ethereum

def test_get_balance():
    ethereum = Ethereum("http://localhost:8545")
    balance = ethereum.get_balance("0x5409ed021d9299bf6814279a6a1411a7e866a631")
    assert balance == 100

def test_send_transaction():
    ethereum = Ethereum("http://localhost:8545")
    tx_hash = ethereum.send_transaction("0x5409ed021d9299bf6814279a6a1411a7e866a631", "0x9c98e073ed711c747546058b6f6564908f63a946", 10)
    assert tx_hash is not None
