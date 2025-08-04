import os
import sys

import web3
from dotenv import load_dotenv
from solcx import compile_source
from web3.auto.infura import w3

load_dotenv()

# Connect to the Ethereum network
w3 = w3(web3.HTTPProvider(os.getenv("INFURA_URL")))

# Compile the smart contract source code
contract_source_code = """
pragma solidity ^0.8.0;

contract Bank {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        msg.sender.transfer(amount);
    }
}
"""
compiled_sol = compile_source(contract_source_code)
contract_interface = compiled_sol["<stdin>:Bank"]

# Deploy the smart contract
w3.eth.defaultAccount = w3.eth.accounts[0]
Bank = w3.eth.contract(
    abi=contract_interface["abi"], bytecode=contract_interface["bin"]
)
tx_hash = Bank.constructor().transact()
tx_receipt = w3.eth.waitForTransactionReceipt(tx_hash)
bank_address = tx_receipt.contractAddress

# Get the contract instance
Bank = w3.eth.contract(address=bank_address, abi=contract_interface["abi"])

# Interact with the smart contract


def deposit_funds(address, value):
    Bank.functions.deposit().transact({"from": address, "value": value})


def withdraw_funds(address, amount):
    Bank.functions.withdraw(amount).transact({"from": address})


# Example usage
if len(sys.argv) < 2:
    print("Usage: python interact.py [deposit|withdraw] [address] [value|amount]")
    sys.exit(1)

operation = sys.argv[1]
address = sys.argv[2]

if operation == "deposit":
    value = int(sys.argv[3])
    deposit_funds(address, value)
elif operation == "withdraw":
    amount = int(sys.argv[3])
    withdraw_funds(address, amount)
else:
    print("Invalid operation. Use 'deposit' or 'withdraw'.")
    sys.exit(1)
