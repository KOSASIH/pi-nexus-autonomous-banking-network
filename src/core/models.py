# src/core/models.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

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
