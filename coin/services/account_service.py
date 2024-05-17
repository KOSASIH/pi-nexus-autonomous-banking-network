# services/account_service.py
from inject import inject
from models.account import Account


class AccountService:
    @inject
    def __init__(self, account_repository):
        self.account_repository = account_repository

    def create_account(self, username: str) -> Account:
        account = Account(username=username)
        self.account_repository.save(account)
        return account

    def get_account(self, username: str) -> Account:
        return self.account_repository.get_by_username(username)
