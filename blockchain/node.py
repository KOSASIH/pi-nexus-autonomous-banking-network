import hashlib
import json
import time
from datetime import datetime
from typing import List, Union

from blockchain.block import Block
from blockchain.transaction import Transaction
from security.encryption import encrypt, decrypt

class Node:
    def __init__(self, identifier: str, private_key: str, public_key: str, ledger: List[Block], chain_head_identifier: str = None):
        self.identifier = identifier
        self.private_key = private_key
        self.public_key = public_key
        self.ledger = ledger
        self.chain_head_identifier = chain_head_identifier

    def __repr__(self):
        return f"Node({self.identifier}, {self.ledger}, {self.chain_head_identifier})"

    def update_chain_head(self, new_chain_head_identifier: str):
        self.chain_head_identifier = new_chain_head_identifier

    def create_new_block(self, data: Union[str, dict]):
        new_block = Block(
            identifier=len(self.ledger),
            previous_hash=self.ledger[-1].hash if self.ledger else None,
            data=data
        )
        self.ledger.append(new_block)
        self.update_chain_head(new_block.identifier)

    def update_ledger(self, new_ledger: List[Block]):
        self.ledger = new_ledger

    def create_new_transaction(self, sender: str, receiver: str, amount: float, message: str = None):
        transaction = Transaction(
            sender=sender,
            receiver=receiver,
            amount=amount,
            message=message,
            timestamp=int(time.time()),
            signature=encrypt(message, self.private_key)
        )
        return transaction

    def validate_transaction(self, transaction: Transaction):
        if transaction.sender == transaction.receiver:
            return False
        if not decrypt(transaction.signature, self.public_key) == transaction.message:
            return False
        if not self.check_balance(transaction.sender, transaction.amount):
            return False
        return True

    def check_balance(self, account: str, amount: float):
        balance = 0
        for block in self.ledger:
            for transaction in block.transactions:
                if transaction.sender == account:
                    balance -= transaction.amount
                elif transaction.receiver == account:
                    balance += transaction.amount
        return balance >= amount

    def get_transaction_history(self, account: str):
        transaction_history = []
        for block in self.ledger:
            for transaction in block.transactions:
                if transaction.sender == account or transaction.receiver == account:
                    transaction_history.append(transaction)
        return transaction_history

    def get_balance(self, account: str):
        balance = 0
        for block in self.ledger:
            for transaction in block.transactions:
                if transaction.sender == account:
                    balance -= transaction.amount
                elif transaction.receiver == account:
                    balance += transaction.amount
        return balance

    def get_chain_head(self):
        return self.ledger[self.chain_head_identifier]

    def get_chain_head_hash(self):
        return self.get_chain_head().hash

    def get_chain_head_timestamp(self):
        return self.get_chain_head().timestamp

    def get_chain_head_data(self):
        return self.get_chain_head().data

    def get_chain_head_previous_hash(self):
        return self.get_chain_head().previous_hash

    def get_chain_head_transactions(self):
        return self.get_chain_head().transactions

    def get_chain_head_block_identifier(self):
        return self.get_chain_head().identifier

    def get_chain_head_block_hash(self):
        return self.get_chain_head().hash

    def get_chain_head_block_timestamp(self):
        return self.get_chain_head().timestamp

    def get_chain_head_block_data(self):
        return self.get_chain_head().data

    def get_chain_head_block_previous_hash(self):
        return self.get_chain_head().previous_hash

    def get_chain_head_block_transactions(self):
        return self.get_chain_head().transactions

    def get_chain_head_block_transactions_count(self):
        return len(self.get_chain_head().transactions)

    def get_chain_head_block_transactions_total_amount(self):
        total_amount = 0
        for transaction in self.get_chain_head().transactions:
            total_amount += transaction.amount
        return total_amount

    def get_chain_head_block_transactions_total_fees(self):
        total_fees = 0
        for transaction in self.get_chain_head().transactions:
            total_fees += transaction.fee
        return total_fees

    def get_chain_head_block_transactions_total_count(self):
        total_count = 0
        for transaction in self.get_chain_head().transactions:
            total_count += 1
        return total_count

    def get_chain_head_block_transactions_total_size(self):
        total_size = 0
        for transaction in self.get_chain_head().transactions:
            total_size += len(transaction.to_json())
        return total_size

    def get_chain_head_block_transactions_total_time(self):
        total_time = 0
        for transaction in self.get_chain_head().transactions:
            total_time += transaction.time
        return total_time

    def get_chain_head_block_transactions_total_gas(self):
        total_gas = 0
        for transaction in self.get_chain_head().transactions:
            total_gas += transaction.gas
        return total_gas

    def get_chain_head_block_transactions_total_gas_price(self):
        total_gas_price = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_price += transaction.gas_price
        return total_gas_price

    def get_chain_head_block_transactions_total_gas_used(self):
        total_gas_used = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_used += transaction.gas_used
        return total_gas_used

    def get_chain_head_block_transactions_total_gas_refund(self):
        total_gas_refund = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_refund += transaction.gas_refund
        return total_gas_refund

    def get_chain_head_block_transactions_total_gas_fee(self):
        total_gas_fee = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_fee += transaction.gas_fee
        return total_gas_fee

    def get_chain_head_block_transactions_total_gas_tip(self):
        total_gas_tip = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_tip += transaction.gas_tip
        return total_gas_tip

    def get_chain_head_block_transactions_total_gas_base(self):
        total_gas_base = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_base += transaction.gas_base
        return total_gas_base

    def get_chain_head_block_transactions_total_gas_price_wei(self):
        total_gas_price_wei = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_price_wei += transaction.gas_price_weireturn total_gas_price_wei

    def get_chain_head_block_transactions_total_gas_used_wei(self):
        total_gas_used_wei = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_used_wei += transaction.gas_used_wei
        return total_gas_used_wei

    def get_chain_head_block_transactions_total_gas_fee_wei(self):
        total_gas_fee_wei = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_fee_wei += transaction.gas_fee_wei
        return total_gas_fee_wei

    def get_chain_head_block_transactions_total_gas_tip_wei(self):
        total_gas_tip_wei = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_tip_wei += transaction.gas_tip_wei
        return total_gas_tip_wei

    def get_chain_head_block_transactions_total_gas_base_wei(self):
        total_gas_base_wei = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_base_wei += transaction.gas_base_wei
        return total_gas_base_wei

    def get_chain_head_block_transactions_total_gas_price_ether(self):
        total_gas_price_ether = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_price_ether += transaction.gas_price_ether
        return total_gas_price_ether

    def get_chain_head_block_transactions_total_gas_used_ether(self):
        total_gas_used_ether = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_used_ether += transaction.gas_used_ether
        return total_gas_used_ether

    def get_chain_head_block_transactions_total_gas_fee_ether(self):
        total_gas_fee_ether = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_fee_ether += transaction.gas_fee_ether
        return total_gas_fee_ether

    def get_chain_head_block_transactions_total_gas_tip_ether(self):
        total_gas_tip_ether = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_tip_ether += transaction.gas_tip_ether
        return total_gas_tip_ether

    def get_chain_head_block_transactions_total_gas_base_ether(self):
        total_gas_base_ether = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_base_ether += transaction.gas_base_ether
        return total_gas_base_ether

    def get_chain_head_block_transactions_total_gas_price_gwei(self):
        total_gas_price_gwei = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_price_gwei += transaction.gas_price_gwei
        return total_gas_price_gwei

    def get_chain_head_block_transactions_total_gas_used_gwei(self):
        total_gas_used_gwei = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_used_gwei += transaction.gas_used_gwei
        return total_gas_used_gwei

    def get_chain_head_block_transactions_total_gas_fee_gwei(self):
        total_gas_fee_gwei = 0
        for transaction in self.get_chain_head().transactions:
            total_gas_fee_gwei += transaction.gas_fee_gwei
        return total_gas_fee_gwei

    def get_chain_head_block_transactions_total_gas_tip_gwei(self):
        total_gas_tip_gwei = 0
for transaction in self.get_chain_head().transactions:
            total_gas_fee_gwei += transaction.gas_fee_gwei
        return total_gas_fee_gwei
