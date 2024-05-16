# tests/test_blockchain.py

import pytest
from .block import Block
from .blockchain import Blockchain

def test_blockchain_initialization():
    """
    Test the initialization of a new blockchain.
    """
    blockchain = Blockchain()
    assert blockchain.chain[0].index == 0
    assert blockchain.chain[0].previous_hash == "0" * 64
    assert blockchain.chain[0].transactions == []
    assert blockchain.chain[0].hash is not None

def test_blockchain_add_transaction():
    """
    Test the addition of a new transaction to the blockchain.
    """
    blockchain = Blockchain()
    blockchain.add_transaction({"sender": "Alice", "recipient": "Bob", "amount": 100})
    assert blockchain.pending_transactions[0] == {"sender": "Alice", "recipient": "Bob", "amount": 100}

def test_blockchain_mine_block():
    """
    Test the mining of a new block.
    """
    blockchain = Blockchain()
    blockchain.add_transaction({"sender": "Alice", "recipient": "Bob", "amount": 100})
    blockchain.mine_block()
    assert blockchain.chain[-1].index == 1
    assert blockchain.chain[-1].previous_hash == blockchain.chain[0].hash
    assert blockchain.chain[-1].transactions == [{"sender": "Alice", "recipient": "Bob", "amount": 100}]
    assert blockchain.chain[-1].hash is not None

def test_blockchain_proof_of_work():
    """
    Test the proof-of-work (PoW) algorithm.
    """
    blockchain = Blockchain()
    block = Block(1, "previous_hash", [{"sender": "Alice", "recipient": "Bob", "amount": 100}])
    block_hash = blockchain.proof_of_work(block)
    assert block_hash.startswith("0" * blockchain.difficulty)

def test_blockchain_validate_chain():
    """
    Test the validation of the blockchain.
    """
    blockchain = Blockchain()
    blockchain.mine_block()
    assert blockchain.validate_chain() is True

def test_blockchain_to_dict():
    """
    Test the conversion of the blockchain to a dictionary representation.
    """
    blockchain = Blockchain()
    blockchain_dict = blockchain.to_dict()
    assert blockchain_dict["chain"][0]["index"] == 0
    assert blockchain_dict["chain"][0]["previous_hash"] == "0" * 64
    assert blockchain_dict["chain"][0]["transactions"] == []
    assert blockchain_dict["chain"][0]["hash"] is not None
    assert blockchain_dict["pending_transactions"] == []

def test_blockchain_save_to_file():
    """
    Test the saving of the blockchain to a file.
    """
    blockchain = Blockchain()
    blockchain.mine_block()
    blockchain.save_to_file("test_blockchain.json")
    with open("test_blockchain.json", "r") as f:
        blockchain_dict = json.load(f)
    assert blockchain_dict["chain"][1]["index"] == 1
    assert blockchain_dict["chain"][1]["previous_hash"] == blockchain.chain[0].hash
    assert blockchain_dict["chain"][1]["transactions"] == []
    assert blockchain_dict["chain"][1]["hash"] is not None
    assert blockchain_dict["pending_transactions"] == []
    os.remove("test_blockchain.json")

def test_blockchain_load_from_file():
    """
    Test the loading of a blockchain from a file.
    """
    blockchain = Blockchain()
    blockchain.mine_block()
    blockchain.save_to_file("test_blockchain.json")
    loaded_blockchain = Blockchain.load_from_file("test_blockchain.json")
    assert loaded_blockchain.chain[1] == blockchain.chain[1]
    assert loaded_blockchain.pending_transactions == blockchain.pending_transactions
    os.remove("test_blockchain.json")
