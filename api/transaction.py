from flask import Blueprint, request, jsonify
from . import db
from .models import Transaction
from .serializers import transaction_schema, transactions_schema
from .exceptions import InvalidTransactionError

transactions_api = Blueprint("transactions_api", __name__)

@transactions_api.route("/transactions", methods=["POST"])
def create_transaction():
    data = request.get_json()
    try:
        transaction = Transaction.from_dict(data)
        db.session.add(transaction)
        db.session.commit()
        return transaction_schema.jsonify(transaction)
    except InvalidTransactionError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@transactions_api.route("/transactions", methods=["GET"])
def get_transactions():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    transactions = Transaction.query.paginate(page, per_page)
    return transactions_schema.jsonify(transactions.items), 200

@transactions_api.route("/transactions/<int:transaction_id>", methods=["GET"])
def get_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)
    return transaction_schema.jsonify(transaction), 200

@transactions_api.route("/transactions/<int:transaction_id>", methods=["PUT"])
def update_transaction(transaction_id):
    data = request.get_json()
    transaction = Transaction.query.get_or_404(transaction_id)
    try:
        transaction.update_from_dict(data)
        db.session.commit()
        return transaction_schema.jsonify(transaction), 200
    except InvalidTransactionError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@transactions_api.route("/transactions/<int:transaction_id>", methods=["DELETE"])
def delete_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)
    try:
        db.session.delete(transaction)
        db.session.commit()
        return "", 204
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@transactions_api.errorhandler(InvalidTransactionError)
def handle_invalid_transaction_error(error):
    return jsonify({"error": str(error)}), 400
