# bank_integration/__init__.py
from .bank_api import BankAPI
from .transaction_processor import TransactionProcessor


# bank_integration/bank_api.py
class BankAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_account_balance(self, account_number: str) -> float:
        # implementation
        pass


# bank_integration/transaction_processor.py
class TransactionProcessor:
    def __init__(self, bank_api: BankAPI):
        self.bank_api = bank_api

    def process_transaction(self, transaction: dict) -> bool:
        # implementation
        pass
