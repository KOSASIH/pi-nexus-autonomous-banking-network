from flask import jsonify, request
from flask_login import login_required

from ..models import Transaction
from ..schemas import TransactionSchema
from . import api_bp

# Initialize the transaction schema
transaction_schema = TransactionSchema()
transactions_schema = TransactionSchema(many=True)


@api_bp.route("/transactions", methods=["GET"])
@login_required
def get_transactions():
    """
    Get all transactions.
    """
    transactions = Transaction.query.all()
    return transactions_schema.jsonify(transactions)


@api_bp.route("/transactions/<int:transaction_id>", methods=["GET"])
@login_required
def get_transaction(transaction_id):
    """
    Get a transaction by ID.
    """
    transaction = Transaction.query.get_or_404(transaction_id)
    return transaction_schema.jsonify(transaction)


@api_bp.route("/transactions", methods=["POST"])
@login_required
def create_transaction():
    """
    Create a new transaction.
    """
    data = request.get_json()
    transaction = Transaction.from_dict(data)
    db.session.add(transaction)
    db.session.commit()
    return transaction_schema.jsonify(transaction), 201


@api_bp.route("/transactions/<int:transaction_id>", methods=["PUT"])
@login_required
def update_transaction(transaction_id):
    """
    Update a transaction by ID.
    """
    transaction = Transaction.query.get_or_404(transaction_id)
    data = request.get_json()
    transaction.update_from_dict(data)
    db.session.commit()
    return transaction_schema.jsonify(transaction)


@api_bp.route("/transactions/<int:transaction_id>", methods=["DELETE"])
@login_required
def delete_transaction(transaction_id):
    """
    Delete a transaction by ID.
    """
    transaction = Transaction.query.get_or_404(transaction_id)
    db.session.delete(transaction)
    db.session.commit()
    return "", 204
