from typing import Dict, List
from services.account_service import AccountService
from services.transaction_service import TransactionService
from views.account_view import render_account_balance, render_transaction_history

account_service = AccountService()
transaction_service = TransactionService()

def deposit(account_number: str, amount: float) -> str:
    account = account_service.get_account(account_number)
    account = account_service.deposit(account, amount)
    transactions = transaction_service.get_transactions(account_number)
    return render_account_balance(account) + render_transaction_history(transactions)

def withdraw(account_number: str, amount: float) -> str:
    account = account_service.get_account(account_number)
    account = account_service.withdraw(account, amount)
    transactions = transaction_service.get_transactions(account_number)
    return render_account_balance(account) + render_transaction_history(transactions)

def transfer(from_account_number: str, to_account_number: str, amount: float) -> str:
    from_account = account_service.get_account(from_account_number)
    to_account = account_service.get_account(to_account_number)
    from_account = account_service.transfer(from_account, to_account, amount)
    transactions = transaction_service.get_transactions(from_account_number)
    return render_account_balance(from_account) + render_transaction_history(transactions)
