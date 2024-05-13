# services/banking_service.py
class BankingService:
    def __init__(self, account_repository):
        self.account_repository = account_repository

    def create_account(self, account_number, balance):
        account = Account(account_number, balance)
        self.account_repository.save(account)
        return account

    def get_account(self, account_number):
        return self.account_repository.get(account_number)
