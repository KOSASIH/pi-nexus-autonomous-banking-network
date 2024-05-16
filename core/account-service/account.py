# account.py
from models.account import Account

class AccountService:
    def __init__(self, repository):
        self.repository = repository

    def create_account(self, account_data):
        account = Account(**account_data)
        self.repository.save(account)
        return account

    def get_account(self, account_id):
        return self.repository.get(account_id)
