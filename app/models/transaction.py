from app import db
from app.models import Account, Currency

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
