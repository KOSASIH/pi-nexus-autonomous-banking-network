from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    account_number = db.Column(db.String(64), unique=True, nullable=False)
    balance = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f"Account(account_number={self.account_number}, balance={self.balance})"
