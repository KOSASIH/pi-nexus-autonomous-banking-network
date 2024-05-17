import json

import pytest
from pi_nexus_blockchain.transaction import Transaction


def test_transaction_initialization():
    # Test the initialization of a new transaction
    transaction = Transaction("Alice", "Bob", 100)
    assert transaction.sender == "Alice"
    assert transaction.recipient == "Bob"
    assert transaction.amount == 100


def test_transaction_to_json():
    # Test the conversion of a transaction to a JSON string
    transaction = Transaction("Alice", "Bob", 100)
    json_string = transaction.to_json()
    transaction_dict = json.loads(json_string)
    assert transaction_dict["sender"] == "Alice"
    assert transaction_dict["recipient"] == "Bob"
    assert transaction_dict["amount"] == 100


def test_transaction_equality():
    # Test the equality of two transactions
    transaction1 = Transaction("Alice", "Bob", 100)
    transaction2 = Transaction("Alice", "Bob", 100)
    transaction3 = Transaction("Alice", "Charlie", 100)
    assert transaction1 == transaction2
    assert transaction1 != transaction3


def test_transaction_validation():
    # Test the validation of a transaction
    transaction = Transaction("Alice", "Bob", 100)
    assert transaction.validate() is True

    # Test the validation of a transaction with a negative amount
    transaction = Transaction("Alice", "Bob", -100)
    assert transaction.validate() is False

    # Test the validation of a transaction with a zero amount
    transaction = Transaction("Alice", "Bob", 0)
    assert transaction.validate() is False

    # Test the validation of a transaction with a non-string sender
    transaction = Transaction(123, "Bob", 100)
    assert transaction.validate() is False

    # Test the validation of a transaction with a non-string recipient
    transaction = Transaction("Alice", 123, 100)
    assert transaction.validate() is False

    # Test the validation of a transaction with a non-numeric amount
    transaction = Transaction("Alice", "Bob", "100")
    assert transaction.validate() is False

    # Test the validation of a transaction with a negative fee
    transaction = Transaction("Alice", "Bob", 100, fee=-10)
    assert transaction.validate() is False

    # Test the validation of a transaction with a zero fee
    transaction = Transaction("Alice", "Bob", 100, fee=0)
    assert transaction.validate() is False

    # Test the validation of a transaction with a non-numeric fee
    transaction = Transaction("Alice", "Bob", 100, fee="10")
    assert transaction.validate() is False
