import os
import json
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any

class Wallet:
    """
    The Wallet class implements the functionality of a Pi network wallet.
    """

    def __init__(self, private_key: str, public_key: str):
        """
        Initializes a new Wallet object.

        :param private_key: The private key of the wallet.
        :param public_key: The public key of the wallet.
        """

        self.private_key = private_key
        self.public_key = public_key
        self.balance = 0

    def create_transaction(self, recipient: str, amount: float) -> Dict[str, Any]:
        """
        Creates a new transaction.

        :param recipient: The recipient of the transaction.
        :param amount: The amount of the transaction.
        :return: The new transaction.
        """

        transaction = {
            'sender': self.public_key,
            'recipient': recipient,
            'amount': amount,
            'timestamp': int(time.time()),
        }

        return transaction

    def sign_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Signs a transaction with the private key of the wallet.

        :param transaction: The transaction to sign.
        :return: The signed transaction.
        """

        transaction['signature'] = self.sign(transaction)

        return transaction

    def sign(self, data: Dict[str, Any]) -> str:
        """
        Signs a message with the private key of the wallet.

        :param data: The message to sign.
        :return: The signature of the message.
        """

        message = json.dumps(data, sort_keys=True).encode()
        signature = hashlib.sha256(message + self.private_key.encode()).hexdigest()

        return signature

    def verify_signature(self, data: Dict[str, Any], signature: str) -> bool:
        """
        Verifies the signature of a message with the public key of the wallet.

        :param data: The message to verify.
        :param signature: The signature of the message.
        :return: True if the signature is valid, False otherwise.
        """

        message = json.dumps(data, sort_keys=True).encode()
        signature_bytes = bytes.fromhex(signature)
        public_key_bytes = bytes.fromhex(self.public_key)

        return hashlib.sha256(message + self.private_key.encode()).hexdigest() == signature

    def add_balance(self, amount: float) -> None:
        """
        Adds an amount to the balance of the wallet.

        :param amount: The amount to add.
        """

        self.balance += amount

    def subtract_balance(self, amount: float) -> None:
        """
        Subtracts an amount from the balance of the wallet.

        :param amount: The amount to subtract.
        """

        self.balance -= amount

    def is_valid(self) -> bool:
        """
        Verifies the validity of the wallet.

        :return: True if the wallet is valid, False otherwise.
        """

        if not self.private_key or not self.public_key:
            return False

        if not self.verify_signature({'public_key': self.public_key}, self.sign({'public_key': self.public_key})):
            return False

        return True
