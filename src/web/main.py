# src/web/main.py
from flask import Flask, jsonify, request

from src.core.models import BankAccount

app = Flask(__name__)


@app.route("/api/bank_accounts", methods=["GET"])
def get_bank_accounts():
    session = create_session()
    bank_accounts = session.query(BankAccount).all()
    session.close()
    return jsonify([ba.to_dict() for ba in bank_accounts])


@app.route("/api/bank_accounts/<int:account_id>", methods=["GET"])
def get_bank_account_by_id(account_id):
    session = create_session()
    bank_account = session.query(BankAccount).filter_by(id=account_id).first()
    session.close()
    if bank_account:
        return jsonify(bank_account.to_dict())
    return jsonify({"error": "Bank account not found"}), 404


@app.route("/api/bank_accounts", methods=["POST"])
def create_bank_account():
    data = request.get_json()
    account_number = data.get("account_number")
    account_holder = data.get("account_holder")

    if not account_number or not account_holder:
        return jsonify({"error": "Missing account number or account holder"}), 400

    session = create_session()
    new_account = BankAccount.create(session, account_number, account_holder)
    session.close()

    return jsonify(new_account.to_dict()), 201


@app.route("/api/bank_accounts/<int:account_id>", methods=["PUT"])
def update_bank_account(account_id):
    data = request.get_json()
    account_number = data.get("account_number")
    account_holder = data.get("account_holder")

    if not account_number or not account_holder:
        return jsonify({"error": "Missing account number or account holder"}), 400

    session = create_session()
    updated_account = BankAccount.update(
        session, account_id, account_number, account_holder
    )
    session.close()

    if updated_account:
        return jsonify(updated_account.to_dict())
    return jsonify({"error": "Bank account not found"}), 404


@app.route("/api/bank_accounts/<int:account_id>", methods=["DELETE"])
def delete_bank_account(account_id):
    session = create_session()
    success = BankAccount.delete(session, account_id)
    session.close()

    if success:
        return jsonify({"message": "Bank account deleted"})
    return jsonify({"error": "Bank account not found"}), 404


@app.route("/api/bank_accounts/search", methods=["GET"])
def search_bank_accounts():
    query = request.args.get("query")
    session = create_session()
    bank_accounts = BankAccount.get_by_query(session, query)
    session.close()
    return jsonify([ba.to_dict() for ba in bank_accounts])
