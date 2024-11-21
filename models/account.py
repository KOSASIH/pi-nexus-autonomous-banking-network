# models/account.py

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from config.config import DB_BASE_MODEL


class Account(DB_BASE_MODEL):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    number = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    balance = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Transaction(DB_BASE_MODEL):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    amount = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    account = relationship("Account", back_populates="transactions")


class TransferTransaction(DB_BASE_MODEL):
    __tablename__ = "transfer_transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_from_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    account_to_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    amount = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    account_from = relationship(
        "Account",
        foreign_keys=[account_from_id],
        back_populates="transfer_transactions_from",
    )
    account_to = relationship(
        "Account",
        foreign_keys=[account_to_id],
        back_populates="transfer_transactions_to",
    )


class AccountTransferTransaction(DB_BASE_MODEL):
    __tablename__ = "account_transfer_transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    transfer_transaction_id = Column(
        Integer, ForeignKey("transfer_transactions.id"), nullable=False
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    account = relationship("Account", back_populates="account_transfer_transactions")
    transfer_transaction = relationship(
        "TransferTransaction", back_populates="account_transfer_transactions"
    )


class Account(DB_BASE_MODEL):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    number = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    balance = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    transactions = relationship(
        "Transaction", back_populates="account", cascade="all, delete-orphan"
    )
    transfer_transactions_from = relationship(
        "TransferTransaction",
        foreign_keys=[TransferTransaction.account_from_id],
        back_populates="account_from",
    )
    transfer_transactions_to = relationship(
        "TransferTransaction",
        foreign_keys=[TransferTransaction.account_to_id],
        back_populates="account_to",
    )
    account_transfer_transactions = relationship(
        "AccountTransferTransaction",
        back_populates="account",
        cascade="all, delete-orphan",
    )


class Account:
    def __init__(self, account_number, balance):
        self.account_number = account_number
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient balance")
        self.balance -= amount
