import pytest
from block_model import Block

from utils import calculate_difficulty, proof_of_work


def test_proof_of_work():
    """Test the proof_of_work function."""
    previous_hash = "abc"
    data = "Hello, world!"
    difficulty = 3

    nonce = proof_of_work(previous_hash, data, difficulty)
    block = Block(index=0, previous_hash=previous_hash, data=data, nonce=nonce)

    assert block.hash.startswith("0" * difficulty)


def test_calculate_difficulty():
    """Test the calculate_difficulty function."""
    previous_block = Block(
        index=0, previous_hash="abc", data="Hello, world!", nonce=123
    )

    assert calculate_difficulty(previous_block, 1, 10) == 1
    assert calculate_difficulty(previous_block, 2, 10) == 1
    assert calculate_difficulty(previous_block, 1, 20) == 1
    assert calculate_difficulty(previous_block, 1, 5) == 2
    assert calculate_difficulty(previous_block, 2, 5) == 1
    assert calculate_difficulty(previous_block, 1, 20) == 0
