from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Coin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    price = db.Column(db.Float, nullable=False)
    market_cap = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)

class Wallet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref=db.backref('wallets', lazy=True))
    coin_id = db.Column(db.Integer, db.ForeignKey('coin.id'))
    coin = db.relationship('Coin', backref=db.backref('wallets', lazy=True))
    balance = db.Column(db.Float, nullable=False)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    wallet_id = db.Column(db.Integer, db.ForeignKey('wallet.id'))
    wallet = db.relationship('Wallet', backref=db.backref('transactions', lazy=True))
    coin_id = db.Column(db.Integer, db.ForeignKey('coin.id'))
    coin = db.relationship('Coin', backref=db.backref('transactions', lazy=True))
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
