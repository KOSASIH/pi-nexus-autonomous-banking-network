import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from database.database import Database
from database.models import User, Account, Transaction

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY")

db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt = JWTManager(app)

logger = logging.getLogger(__name__)

@app.route("/api/users", methods=["GET"])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    user = User.query.get(user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict())

@app.route("/api/users", methods=["POST"])
def create_user():
    data = request.get_json()
    user = User(username=data["username"], password=data["password"], email=data["email"])
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201

@app.route("/api/users/<int:user_id>", methods=["PUT"])
@jwt_required
def update_user(user_id):
    user = User.query.get(user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    data = request.get_json()
    user.username = data["username"]
    user.email = data["email"]
    db.session.commit()
    return jsonify(user.to_dict())

@app.route("/api/users/<int:user_id>", methods=["DELETE"])
@jwt_required
def delete_user(user_id):
    user = User.query.get(user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted"})

@app.route("/api/accounts", methods=["GET"])
def get_accounts():
    accounts = Account.query.all()
    return jsonify([account.to_dict() for account in accounts])

@app.route("/api/accounts/<int:account_id>", methods=["GET"])
def get_account(account_id):
    account = Account.query.get(account_id)
    if account is None:
        return jsonify({"error": "Account not found"}), 404
    return jsonify(account.to_dict())

@app.route("/api/accounts", methods=["POST"])
@jwt_required
def create_account():
    data = request.get_json()
    account = Account(user_id=data["user_id"], account_type=data["account_type"], balance=data["balance"])
    db.session.add(account)
    db.session.commit()
    return jsonify(account.to_dict()), 201

@app.route("/api/accounts/<int:account_id>", methods=["PUT"])
@jwt_required
def update_account(account_id):
    account = Account.query.get(account_id)
    if account is None:
        return jsonify({"error": "Account not found"}), 404
    data = request.get_json()
    account.account_type = data["account_type"]
    account.balance = data["balance"]
    db.session.commit()
    return jsonify(account.to_dict())

@app.route("/api/accounts/<int:account_id>", methods=["DELETE"])
@jwt_required
def delete_account(account_id):
    account = Account.query.get(account_id)
    if account is None:
        return jsonify({"error": "Account not found"}), 404
    db.session.delete(account)
    db.session.commit()
    return jsonify({"message": "Account deleted"})

@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    transactions = Transaction.query.all()
    return jsonify([transaction.to_dict() for transaction in transactions])

@app.route("/api/transactions/<int:transaction_id>", methods=["GET"])
def get_transaction(transaction_id):
    transaction = Transaction.query.get(transaction_id)
    if transaction is None:
        return jsonify({"error": "Transaction not found"}), 404
    return jsonify(transaction.to_dict())

@app.route("/api/transactions", methods=["POST"])
@jwt_required
def create_transaction():
    data = request.get_json()
    transaction = Transaction(account_id=data["account_id"], amount=data["amount"], transaction_type=data["transaction_type"])
    db.session.add(transaction)
    db.session.commit()
    return jsonify(transaction.to_dict()), 201

@app.route("/api/transactions/<int:transaction_id>", methods=["PUT"])
@jwt_required
def update_transaction(transaction_id):
    transaction = Transaction.query.get(transaction_id)
    if transaction is None:
        return jsonify({"error": "Transaction not found"}), 404
    data = request.get_json()
    transaction.amount = data["amount"]
    transaction.transaction_type = data["transaction_type"]
    db.session.commit()
    return jsonify(transaction.to_dict())

@app.route("/api/transactions/<int:transaction_id>", methods=["DELETE"])
@jwt_required
def delete_transaction(transaction_id):
    transaction = Transaction.query.get(transaction_id)
    if transaction is None:
        return jsonify({"error": "Transaction not found"}), 404
    db.session.delete(transaction)
    db.session.commit()
    return jsonify({"message": "Transaction deleted"})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data["username"]
    password = data["password"]
    # Implement authentication logic here
    if username == "admin" and password == "password":
        access_token = create_access_token(identity=username)
        return jsonify({"access_token": access_token})
    return jsonify({"error": "Invalid credentials"}), 401

if __name__ == "__main__":
    app.run(debug=True)
