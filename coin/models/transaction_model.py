from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    wallet_id = Column(Integer, ForeignKey('wallets.id'))
    wallet = relationship('Wallet', backref='transactions')
    coin_id = Column(Integer, ForeignKey('coins.id'))
    coin = relationship('Coin', backref='transactions')
    amount = Column(Integer)
    timestamp = Column(DateTime)
