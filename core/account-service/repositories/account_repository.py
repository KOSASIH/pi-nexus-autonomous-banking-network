# account_repository.py
from models.account import Account


class AccountRepository:
    def __init__(self, db):
        self.db = db

    def save(self, account):
        self.db.session.add(account)
        self.db.session.commit()

    def get(self, account_id):
        return self.db.session.query(Account).get(account_id)
