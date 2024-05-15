import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from wallet import Wallet

# Create a new wallet
wallet = Wallet(
    private_key="0x1234567890abcdef1234567890abcdef1234567890abcdef",
    public_key="0x0000000000000000000000000000000000000000000000000000000000000000",
)

# Create a new transaction
transaction = wallet.create_transaction(
    recipient="0x0000000000000000000000000000000000000001", amount=10
)

# Sign the transaction with the private key of the wallet
signed_transaction = wallet.sign_transaction(transaction)

# Verify the signature of the transaction with the public key of the wallet
print(wallet.verify_signature(signed_transaction, signed_transaction["signature"]))

# Add the balance of the transaction to the balance of the wallet
wallet.add_balance(signed_transaction["amount"])

# Subtract the balance of the transaction from the balance of the wallet
wallet.subtract_balance(signed_transaction["amount"])

# Verify the validity of the wallet
print(wallet.is_valid())
