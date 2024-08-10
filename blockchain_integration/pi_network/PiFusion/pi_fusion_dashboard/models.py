from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Node(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    incentivization_type = db.Column(db.String(50), nullable=False)
    reputation = db.Column(db.Float, nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('node.id'))
    parent = db.relationship('Node', remote_side=[id])

    def __repr__(self):
        return f'<Node {self.name}>'

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    node_id = db.Column(db.Integer, db.ForeignKey('node.id'))
    node = db.relationship('Node', backref=db.backref('transactions', lazy=True))

    def __repr__(self):
        return f'<Transaction {self.type}>'
