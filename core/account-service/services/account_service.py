# account_service.py
from account import AccountService
from repositories.account_repository import AccountRepository

class AccountServiceImpl(AccountService):
    def __init__(self, db):
        repository = AccountRepository(db)
        super().__init__(repository)
