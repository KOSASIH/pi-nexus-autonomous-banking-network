import pytest
from blockchain.pi_network import PiNetwork

def test_get_balance():
    pi_network = PiNetwork("http://localhost:8000")
    balance = pi_network.get_balance("pi1234567890abcdefghij")
    assert balance == 100000000000000000000

def test_send_transaction():
    pi_network = PiNetwork("http://localhost:8000")
    tx_hash = pi_network.send_transaction("pi1234567890abcdefghij", "pi0987654321klmnopqrst", 1000000000000000000)
    assert tx_hash is not None
