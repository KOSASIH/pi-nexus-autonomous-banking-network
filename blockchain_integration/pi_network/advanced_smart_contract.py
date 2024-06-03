import json
import os
from typing import Dict, List, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from web3 import Web3
from web3.contract import Contract
from web3.middleware import geth_poa_middleware
from web3.providers import HTTPProvider

# Configuration and constants
CHAIN_ID = os.environ.get("CHAIN_ID")
NODE_URL = os.environ.get("NODE_URL")
CONTRACT_ADDRESS = os.environ.get("CONTRACT_ADDRESS")
PRIVATE_KEY = os.environ.get("PRIVATE_KEY")

# Web3 provider
w3 = Web3(HTTPProvider(NODE_URL))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Load the smart contract ABI
with open("pi_nexus_abi.json", "r") as f:
    abi = json.load(f)

# Load the smart contract bytecode
with open("pi_nexus_bytecode.bin", "rb") as f:
    bytecode = f.read()

# Create a new smart contract instance
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)


# Define a function to generate a new key pair
def generate_key_pair() -> Tuple[str, str]:
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    private_key = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_key = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        .decode("utf-8")
    )
    return private_key, public_key


# Define a function to encrypt data using the public key
def encrypt_data(public_key: str, data: str) -> str:
    key = serialization.load_ssh_public_key(
        public_key.encode("utf-8"), backend=default_backend()
    )
    encrypted_data = key.encrypt(
        data.encode("utf-8"),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return encrypted_data.decode("utf-8")


# Define a function to decrypt data using the private key
def decrypt_data(private_key: str, encrypted_data: str) -> str:
    key = serialization.load_pem_private_key(
        private_key.encode("utf-8"), password=None, backend=default_backend()
    )
    decrypted_data = key.decrypt(
        encrypted_data.encode("utf-8"),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return decrypted_data.decode("utf-8")


# Define a function to create a new account
def create_account() -> str:
    private_key, public_key = generate_key_pair()
    tx_hash = contract.functions.createAccount(public_key).transact(
        {"from": w3.eth.accounts[0]}
    )
    w3.eth.waitForTransactionReceipt(tx_hash)
    return private_key


# Define a function to deposit funds into an account
def deposit_funds(account_id: str, amount: int) -> str:
    tx_hash = contract.functions.depositFunds(account_id, amount).transact(
        {"from": w3.eth.accounts[0]}
    )
    w3.eth.waitForTransactionReceipt(tx_hash)
    return tx_hash


# Define a function to transfer funds between accounts
def transfer_funds(sender_id: str, recipient_id: str, amount: int) -> str:
    tx_hash = contract.functions.transferFunds(
        sender_id, recipient_id, amount
    ).transact({"from": w3.eth.accounts[0]})
    w3.eth.waitForTransactionReceipt(tx_hash)
    return tx_hash


# Define a function to get the balance of an account
def get_balance(account_id: str) -> int:
    return contract.functions.getBalance(account_id).call()


# Define a function to get the transaction history of an account
def get_transaction_history(account_id: str) -> List[Dict]:
    return contract.functions.getTransactionHistory(account_id).call()


if __name__ == "__main__":
    # Create a new account
    account_private_key = create_account()
    account_public_key = w3.eth.account.privateKeyToAccount(
        account_private_key
    ).publicKey

    # Deposit funds into the account
    deposit_funds(account_public_key, 100)

    # Transfer funds to another account
    recipient_public_key = "..."
    transfer_funds(account_public_key, recipient_public_key, 50)

    # Get the balance of the account
    balance = get_balance(account_public_key)
    print(f"Account balance: {balance}")

    # Get the transaction history of the account
    transaction_history = get_transaction_history(account_public_key)
    print(f"Account transaction history: {transaction_history}")
