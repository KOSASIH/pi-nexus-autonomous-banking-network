# services/__init__.py
from inject import Binder, Module, inject
from .account_service import AccountService
from .transaction_service import TransactionService
from .user_service import UserService

class AppModule(Module):
    @inject
    def __init__(self, account_service: AccountService, transaction_service: TransactionService, user_service: UserService):
        self.account_service = account_service
        self.transaction_service = transaction_service
        self.user_service = user_service

def configure(binder: Binder):
    binder.bind(AccountService, AccountService)
    binder.bind(TransactionService,
