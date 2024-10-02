from flask import Blueprint, jsonify, request
from flask_sqlalchemy import sqlalchemy

from web.models import Transaction

transactions_bp = Blueprint("transactions", __name__, url_prefix="/api/transactions")


@transactions_bp.route("", methods=["GET"])
@jwt_required()
def get_transactions():
    transactions = Transaction.query.all()

    return jsonify([transaction.to_dict() for transaction in transactions]), 200


@transactions_bp.route("/<int:transaction_id>", methods=["GET"])
@jwt_required()
def get_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)

    return jsonify(transaction.to_dict()), 200


@transactions_bp.route("", methods=["POST"])
@jwt_required()
def create_transaction():
    amount = request.json.get("amount")
    sender_id = request.json.get("sender_id")
    receiver_id = request.json.get("receiver_id")

    if not amount or not sender_id or not receiver_id:
        return jsonify({"error": "Missing amount or sender_id or receiver_id"}), 400

    sender = User.query.get_or_404(sender_id)
    receiver = User.query.get_or_404(receiver_id)

    transaction = Transaction(amount=amount, sender=sender, receiver=receiver)

    db.session.add(transaction)
    db.session.commit()

    return jsonify(transaction.to_dict()), 201


@transactions_bp.route("/<int:transaction_id>", methods=["PUT"])
@jwt_required()
def update_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)

    amount = request.json.get("amount")
    sender_id = request.json.get("sender_id")
    receiver_id = request.json.get("receiver_id")

    if amount is not None:
        transaction.amount = amount

    if sender_id is not None:
        sender = User.query.get_or_404(sender_id)
        transaction.sender = sender

    if receiver_id is not None:
        receiver = User.query.get_or_404(receiver_id)
        transaction.receiver = receiver

    db.session.commit()

    return jsonify(transaction.to_dict()), 200


@transactions_bp.route("/<int:transaction_id>", methods=["DELETE"])
@jwt_required()
def delete_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)

    db.session.delete(transaction)
    db.session.commit()

    return jsonify({"message": "Transaction deleted"}), 200
