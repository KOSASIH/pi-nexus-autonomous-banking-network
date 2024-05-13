from typing import Dict, List

from views.transaction_view import render_transaction_details, render_transaction_list

from services.transaction_service import TransactionService

transaction_service = TransactionService()


def create_transaction(account_number: str, transaction: Dict) -> str:
    transaction = transaction_service.create_transaction(account_number, transaction)
    transactions = transaction_service.get_transactions(account_number)
    return render_transaction_list(transactions) + render_transaction_details(
        transaction
    )


def update_transaction(transaction_id: str, transaction: Dict) -> str:
    transaction = transaction_service.update_transaction(transaction_id, transaction)
    transactions = transaction_service.get_transactions(transaction["account_number"])
    return render_transaction_list(transactions) + render_transaction_details(
        transaction
    )


def delete_transaction(transaction_id: str) -> str:
    account_number = transaction_service.get_transaction(transaction_id)[
        "account_number"
    ]
    transaction_service.delete_transaction(transaction_id)
    transactions = transaction_service.get_transactions(account_number)
    return render_transaction_list(transactions)
