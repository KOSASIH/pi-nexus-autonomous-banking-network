# tests/test_block.py
import pytest
from blockchain.block import Block

def test_block_init():
    block = Block(0, "0", datetime.datetime.now(), "Genesis Block", "0")
    assert block.index == 0
    assert block.previous_hash == "0"
    # ...
