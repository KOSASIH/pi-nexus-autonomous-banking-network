import pytest
from blockchain.bitcoin import Bitcoin

def test_get_balance():
    bitcoin = Bitcoin("http://localhost:8332")
    balance = bitcoin.get_balance("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2")
    assert balance == 100

def test_send_transaction():
    bitcoin = Bitcoin("http://localhost:8332")
    tx_hash = bitcoin.send_transaction("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "175tWpb8K1S7NmH4Zx6rewF9WQQgnfDoM", 10)
    assert tx_hash is not None
