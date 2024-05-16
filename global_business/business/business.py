# business/business.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Customer:
    id: int
    name: str
    email: str
    balance: float = 0.0


class BankAccount(ABC):
    def __init__(self, customer: Customer):
        self.customer = customer
        self.transactions: List[Dict[str, Any]] = []

    @abstractmethod
    def deposit(self, amount: float) -> None:
        pass

    @abstractmethod
    def withdraw(self, amount: float) -> None:
        pass

    def get_balance(self) -> float:
        return self.customer.balance

    def get_transactions(self) -> List[Dict[str, Any]]:
        return self.transactions


class CheckingAccount(BankAccount):
    def deposit(self, amount: float) -> None:
        self.customer.balance += amount
        self.transactions.append({"type": "deposit", "amount": amount})

    def withdraw(self, amount: float) -> None:
        if amount > self.customer.balance:
            logger.warning("Insufficient funds")
            return
        self.customer.balance -= amount
        self.transactions.append({"type": "withdrawal", "amount": amount})


class SavingsAccount(BankAccount):
    def deposit(self, amount: float) -> None:
        self.customer.balance += amount
        self.transactions.append({"type": "deposit", "amount": amount})

    def withdraw(self, amount: float) -> None:
        if amount > self.customer.balance:
            logger.warning("Insufficient funds")
            return
        self.customer.balance -= amount
        self.transactions.append({"type": "withdrawal", "amount": amount})


class Business:
    def __init__(self):
        self.customers: Dict[int, Customer] = {}
        self.accounts: Dict[int, BankAccount] = {}

    def create_customer(self, name: str, email: str) -> int:
        customer_id = len(self.customers) + 1
        customer = Customer(customer_id, name, email)
        self.customers[customer_id] = customer
        return customer_id

    def create_account(self, customer_id: int, account_type: str) -> int:
        if customer_id not in self.customers:
            logger.error("Customer not found")
            return None
        account_id = len(self.accounts) + 1
        if account_type == "checking":
            account = CheckingAccount(self.customers[customer_id])
        elif account_type == "savings":
            account = SavingsAccount(self.customers[customer_id])
        else:
            logger.error("Invalid account type")
            return None
        self.accounts[account_id] = account
        return account_id

    def deposit(self, account_id: int, amount: float) -> None:
        if account_id not in self.accounts:
            logger.error("Account not found")
            return
        self.accounts[account_id].deposit(amount)

    def withdraw(self, account_id: int, amount: float) -> None:
        if account_id not in self.accounts:
            logger.error("Account not found")
            return
        self.accounts[account_id].withdraw(amount)

    def get_customer(self, customer_id: int) -> Customer:
        return self.customers.get(customer_id)

    def get_account(self, account_id: int) -> BankAccount:
        return self.accounts.get(account_id)
