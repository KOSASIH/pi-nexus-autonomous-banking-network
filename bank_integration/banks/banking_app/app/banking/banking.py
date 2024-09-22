from flask import request, jsonify, current_app
from flask_jwt_extended import jwt_required
from .models import Account
from .schemas import AccountSchema

@banking_blueprint.route("/accounts", methods=["GET"])
@jwt_required
def get_accounts():
    accounts = Account.query.all()
    return jsonify([AccountSchema().dump(account) for account in accounts])

@banking_blueprint.route("/accounts", methods=["POST"])
@jwt_required
def create_account():
    account_number = request.json.get("account_number")
    account = Account(account_number=account_number)
    db.session.add(account)
    db.session.commit()
    return jsonify(message="Account created successfully")

@banking_blueprint.route("/accounts/<int:account_id>", methods=["GET"])
@jwt_required
def get_account(account_id):
    account = Account.query.get(account_id)
    if account:
        return jsonify(AccountSchema().dump(account))
    return jsonify(error="Account not found"), 404

@banking_blueprint.route("/accounts/<int:account_id>", methods=["PUT"])
@jwt_required
def update_account(account_id):
    account = Account.query.get(account_id)
    if account:
        account.account_number = request.json.get("account_number")
        db.session.commit()
        return jsonify(message="Account updated successfully")
    return jsonify(error="Account not found"), 404

@banking_blueprint.route("/accounts/<int:account_id>", methods=["DELETE"])
@jwt_required
def delete_account(account_id):
    account = Account.query.get(account_id)
    if account:
        db.session.delete(account)
        db.session.commit()
        return jsonify(message="Account deleted successfully")
    return jsonify(error="Account not found"), 404
