from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship

# Initialize database
db = SQLAlchemy()

class User(db.Model):
    """
    A User model for the FineX project.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

    # Relationships
    transactions = relationship('Transaction', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

class Transaction(db.Model):
    """
    A Transaction model for the FineX project.
    """
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(16), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

    # Foreign key
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<Transaction {self.id}>'
