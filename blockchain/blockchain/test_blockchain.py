import pytest

from blockchain import Blockchain
from blockchain.block import Block


def test_create_genesis_block():
    """Test the creation of the genesis block."""
    blockchain = Blockchain()
    genesis_block = blockchain.chain[0]

    assert genesis_block.index == 0
    assert genesis_block.previous_hash == "0"
    assert genesis_block.data == "Genesis Block"


def test_add_block():
    """Test adding blocks to the blockchain."""
    blockchain = Blockchain()
    blockchain.add_block("First block")
    blockchain.add_block("Second block")

    assert len(blockchain.chain) == 3
    assert blockchain.chain[1].data == "First block"
    assert blockchain.chain[2].data == "Second block"


def test_is_valid():
    """Test the integrity of the blockchain."""
    blockchain = Blockchain()
    blockchain.add_block("First block")
    blockchain.add_block("Second block")

    assert blockchain.is_valid()

    # Modify the hash of the first block
    blockchain.chain[0].hash = "0" * 64
    assert not blockchain.is_valid()

    # Modify the previous hash of the second block
    blockchain.chain[1].previous_hash = "0" * 64
    assert not blockchain.is_valid()
