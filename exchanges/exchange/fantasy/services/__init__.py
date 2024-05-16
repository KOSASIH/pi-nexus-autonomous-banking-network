# services/__init__.py
from inject import Binder
from .account_service import AccountService
from .transaction_service import TransactionService

def configure(binder: Binder):
    binder.bind(AccountService, AccountService)
    binder.bind(TransactionService, TransactionService)
