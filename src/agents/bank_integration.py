import os
import requests
from blockchain import Blockchain
from consensus import Consensus
from cryptography import generate_private_key, sign_transaction, verify_signature
from interfaces.api import create_transaction, get_balance
from services.analytics import analyze_transactions
from services.monitoring import monitor_node

# Load private key from environment variable or generate a new one
private_key = os.getenv('PRIVATE_KEY') or generate_private_key()

# Create a new blockchain instance
blockchain = Blockchain()

# Create a new consensus instance
consensus = Consensus(blockchain)

# Monitor node and detect any malicious behavior
monitor_node(blockchain, consensus)

# Analyze transactions for anomalies
analyze_transactions(blockchain)

# Get the balance of a bank in the network
def get_bank_balance(bank_id):
    balance = get_balance(bank_id)
    if balance:
        return balance
    else:
        return "Bank not found"

# Create a transaction between two banks in the network
def create_bank_transaction(sender_id, receiver_id, amount):
    if sender_id != receiver_id:
        # Verify sender's private key
        if verify_signature(sender_id, sender_id, private_key):
            # Create a new transaction
            transaction = create_transaction(sender_id, receiver_id, amount)

            # Sign the transaction with the sender's private key
            signed_transaction = sign_transaction(transaction, private_key)

            # Add the signed transaction to the blockchain
            if blockchain.add_transaction(signed_transaction):
                return f"Transaction successful. Amount: {amount}"
            else:
                return "Transaction failed. Insufficient funds or error occurred."
        else:
            return "Transaction failed. Invalid sender's private key."
    else:
        return "Transaction failed. Sender and receiver must be different banks."
