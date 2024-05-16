# tests/test_block.py

import pytest
from .block import Block

def test_block_initialization():
    """
    Test the initialization of a new block.
    """
    block = Block(1, "previous_hash", [{"sender": "Alice", "recipient": "Bob", "amount": 100}])
    assert block.index == 1
    assert block.previous_hash == "previous_hash"
    assert block.transactions == [{"sender": "Alice", "recipient": "Bob", "amount": 100}]
    assert block.timestamp is not None
    assert block.hash is not None

def test_block_calculate_hash():
    """
    Test the calculation of a block's hash.
    """
    block = Block(1, "previous_hash", [{"sender": "Alice", "recipient": "Bob", "amount": 100}])
    assert block.calculate_hash() == block.hash

def test_block_to_dict():
    """
    Test the conversion of a block to a dictionary representation.
    """
    block = Block(1, "previous_hash", [{"sender": "Alice", "recipient": "Bob", "amount": 100}])
    block_dict = block.to_dict()
    assert block_dict["index"] == 1
    assert block_dict["previous_hash"] == "previous_hash"
    assert block_dict["transactions"] == [{"sender": "Alice", "recipient": "Bob", "amount": 100}]
    assert block_dict["timestamp"] is not None
    assert block_dict["hash"] is not None

def test_block_equality():
    """
    Test the equality of two blocks.
    """
    block1 = Block(1, "previous_hash", [{"sender": "Alice", "recipient": "Bob", "amount": 100}])
    block2 = Block(1, "previous_hash", [{"sender": "Alice", "recipient": "Bob", "amount": 100}])
    assert block1 == block2

def test_block_inequality():
    """
    Test the inequality of two blocks with different attributes.
    """
    block1 = Block(1, "previous_hash", [{"sender": "Alice", "recipient": "Bob", "amount": 100}])
    block2 = Block(2, "previous_hash", [{"sender": "Alice", "recipient": "Bob", "amount": 100}])
    assert block1 != block2
