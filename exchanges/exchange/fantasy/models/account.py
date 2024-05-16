# models/account.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    balance = Column(Integer, default=0)

    def __init__(self, username: str, balance: int = 0):
        self.username = username
        self.balance = balance

    def __repr__(self):
        return f"Account(username={self.username}, balance={self.balance})"
