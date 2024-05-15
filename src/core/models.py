# src/core/models.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class BankAccount(Base):
    __tablename__ = "bank_accounts"
    id = Column(Integer, primary_key=True)
    account_number = Column(String)
    account_holder = Column(String)

    def to_dict(self):
        return {
            "id": self.id,
            "account_number": self.account_number,
            "account_holder": self.account_holder,
        }


def create_bank_account(session, account_number, account_holder):
    new_account = BankAccount(
        account_number=account_number, account_holder=account_holder
    )
    session.add(new_account)
    session.commit()
    return new_account


def update_bank_account(session, account_id, account_number, account_holder):
    account = session.query(BankAccount).filter_by(id=account_id).first()
    if account:
        account.account_number = account_number
        account.account_holder = account_holder
        session.commit()
        return account
    return None


def delete_bank_account(session, account_id):
    account = session.query(BankAccount).filter_by(id=account_id).first()
    if account:
        session.delete(account)
        session.commit()
        return True
    return False


def get_bank_account_by_id(session, account_id):
    return session.query(BankAccount).filter_by(id=account_id).first()


def get_bank_accounts_by_query(session, query):
    return (
        session.query(BankAccount)
        .filter(
            BankAccount.account_number.ilike(f"%{query}%")
            | BankAccount.account_holder.ilike(f"%{query}%")
        )
        .all()
    )
