from flask import Blueprint, request, jsonify
from api.models import Wallet
from api.schemas import WalletSchema

wallet_blueprint = Blueprint("wallet", __name__)

@wallet_blueprint.route("/create", methods=["POST"])
def create_wallet():
    data = request.get_json()
    wallet = Wallet(**data)
    db.session.add(wallet)
    db.session.commit()
    return jsonify({"message": "Wallet created successfully"})

@wallet_blueprint.route("/balance", methods=["GET"])
def get_balance():
    wallet_id = request.args.get("wallet_id")
    wallet = Wallet.query.get(wallet_id)
    if wallet:
        return jsonify({"balance": wallet.balance})
    return jsonify({"error": "Wallet not found"}), 404
