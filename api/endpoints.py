from flask import Blueprint, jsonify, request

from .authentication import jwt_required
from .serializers import TransactionSchema, UserSchema

endpoints = Blueprint("endpoints", __name__)


@endpoints.route("/users", methods=["POST"])
def create_user():
    user_schema = UserSchema()
    user = user_schema.load(request.get_json())
    # Create user in the database
    return user_schema.jsonify(user), 201


@endpoints.route("/users/<int:user_id>", methods=["GET"])
@jwt_required()
def get_user(user_id):
    user = get_user_from_database(user_id)
    user_schema = UserSchema()
    return user_schema.jsonify(user)


@endpoints.route("/users/<int:user_id>", methods=["PUT"])
@jwt_required()
def update_user(user_id):
    user = get_user_from_database(user_id)
    user_schema = UserSchema()
    user = user_schema.load(request.get_json(), instance=user)
    # Update user in the database
    return user_schema.jsonify(user)


@endpoints.route("/users/<int:user_id>", methods=["DELETE"])
@jwt_required()
def delete_user(user_id):
    user = get_user_from_database(user_id)
    # Delete user from the database
    return "", 204


@endpoints.route("/transactions", methods=["POST"])
@jwt_required()
def create_transaction():
    transaction_schema = TransactionSchema()
    transaction = transaction_schema.load(request.get_json())
    # Create transaction in the database
    return transaction_schema.jsonify(transaction), 201


@endpoints.route("/transactions", methods=["GET"])
@jwt_required()
def get_transactions():
    transactions = get_transactions_from_database()
    transaction_schema = TransactionSchema(many=True)
    return transaction_schema.jsonify(transactions)


@endpoints.route("/transactions/<int:transaction_id>", methods=["GET"])
@jwt_required()
def get_transaction(transaction_id):
    transaction = get_transaction_from_database(transaction_id)
    transaction_schema = TransactionSchema()
    return transaction_schema.jsonify(transaction)


@endpoints.route("/transactions/<int:transaction_id>", methods=["PUT"])
@jwt_required()
def update_transaction(transaction_id):
    transaction = get_transaction_from_database(transaction_id)
    transaction_schema = TransactionSchema()
    transaction = transaction_schema.load(request.get_json(), instance=transaction)
    # Update transaction in the database
    return transaction_schema.jsonify(transaction)


@endpoints.route("/transactions/<int:transaction_id>", methods=["DELETE"])
@jwt_required()
def delete_transaction(transaction_id):
    transaction = get_transaction_from_database(transaction_id)
    # Delete transaction from the database
    return "", 204
