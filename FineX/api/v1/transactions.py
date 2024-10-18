# transactions.py

from flask import Blueprint, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from marshmallow import Schema, fields, ValidationError

# Initialize the database
db = SQLAlchemy()

# Create a blueprint for transactions
transactions_bp = Blueprint('transactions', __name__)

# Transaction model
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(100), nullable=False)
    recipient = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Transaction schema for validation
class TransactionSchema(Schema):
    sender = fields.Str(required=True)
    recipient = fields.Str(required=True)
    amount = fields.Float(required=True)

transaction_schema = TransactionSchema()
transactions_schema = TransactionSchema(many=True)

# Route to create a new transaction
@transactions_bp.route('/transactions', methods=['POST'])
def create_transaction():
    try:
        # Validate and deserialize input
        data = transaction_schema.load(request.json)
        new_transaction = Transaction(**data)
        db.session.add(new_transaction)
        db.session.commit()
        return transaction_schema.jsonify(new_transaction), 201
    except ValidationError as err:
        return jsonify(err.messages), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to retrieve all transactions
@transactions_bp.route('/transactions', methods=['GET'])
def get_transactions():
    transactions = Transaction.query.all()
    return transactions_schema.jsonify(transactions), 200

# Route to retrieve a specific transaction by ID
@transactions_bp.route('/transactions/<int:transaction_id>', methods=['GET'])
def get_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)
    return transaction_schema.jsonify(transaction), 200

# Route to delete a transaction
@transactions_bp.route('/transactions/<int:transaction_id>', methods=['DELETE'])
def delete_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)
    db.session.delete(transaction)
    db.session.commit()
    return jsonify({"message": "Transaction deleted successfully."}), 204

# Optional: Add any additional transaction-related functionalities here
@transactions_bp.route('/transactions/summary', methods=['GET'])
def transaction_summary():
    total_transactions = Transaction.query.count()
    total_amount = db.session.query(db.func.sum(Transaction.amount)).scalar() or 0
    return jsonify({
        "total_transactions": total_transactions,
        "total_amount": total_amount
    }), 200

# Register the blueprint in the main application
def init_app(app):
    app.register_blueprint(transactions_bp, url_prefix='/api/v1')
