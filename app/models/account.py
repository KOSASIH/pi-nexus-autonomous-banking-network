from datetime import datetime
from flask_login import UserMixin
from app import db, login

class Account(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    balance = db.Column(db.Float, nullable=False, default=0.0)
    currency_id = db.Column(db.Integer, db.ForeignKey('currency.id'), nullable=False)
    transactions = db.relationship('Transaction', backref='account', lazy=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Account {self.username}>'

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('account.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('account.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    currency_id = db.Column(db.Integer, db.ForeignKey('currency.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    sender = db.relationship('Account', foreign_keys=[sender_id])
    receiver = db.relationship('Account', foreign_keys=[receiver_id])
    currency = db.relationship('Currency')

    def __repr__(self):
        return f'<Transaction {self.id}>'

@login.user_loader
def load_user(user_id):
    return Account.query.get(int(user_id))
