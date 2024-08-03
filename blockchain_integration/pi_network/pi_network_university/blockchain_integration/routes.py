# blockchain_integration/routes.py
from flask import Blueprint, request, jsonify
from. import app, db, jwt, bcrypt, w3
from.models import User, Wallet, Transaction
from.schemas import UserSchema, WalletSchema, TransactionSchema
from. import login_manager
import eth_account

user_routes = Blueprint("user_routes", __name__)
wallet_routes = Blueprint("wallet_routes", __name__)
transaction_routes = Blueprint("transaction_routes", __name__)

@user_routes.route("/users", methods=["GET"])
@login_required
def get_users():
    users = User.query.all()
    user_schema = UserSchema(many=True)
    return jsonify(user_schema.dump(users))

@user_routes.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()
    user = User(username=data["username"], email=data["email"])
    user.set_password(data["password"])
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created successfully"})

@wallet_routes.route("/wallets", methods=["GET"])
@login_required
def get_wallets():
    wallets = Wallet.query.all()
    wallet_schema = WalletSchema(many=True)
    return jsonify(wallet_schema.dump(wallets))

@wallet_routes.route("/wallets", methods=["POST"])
@login_required
def create_wallet():
    data = request.get_json()
    wallet = Wallet(user_id=current_user.id, address=data["address"], private_key=data["private_key"])
    db.session.add(wallet)
    db.session.commit()
    return jsonify({"message": "Wallet created successfully"})

@transaction_routes.route("/transactions", methods=["GET"])
@login_required
def get_transactions():
    transactions = Transaction.query.all()
    transaction_schema = TransactionSchema(many=True)
    return jsonify(transaction_schema.dump(transactions))

@transaction_routes.route("/transactions", methods=["POST"])
@login_required
def create_transaction():
    data = request.get_json()
    wallet = Wallet.query.get(data["wallet_id"])
    if wallet:
        tx = w3.eth.account.sign_transaction({
            "nonce": w3.eth.get_transaction_count(wallet.address),
            "gasPrice": w3.utils.to_wei(data["gas_price"], "gwei"),
            "gas": data["gas"],
            "to": data["to_address"],
            "value": w3.utils.to_wei(data["value"], "ether"),
            "data": ""
        }, wallet.private_key)
        tx_hash = w3.eth.send_raw_transaction(tx.rawTransaction)
        transaction = Transaction(wallet_id=wallet.id, tx_hash=tx_hash, from_address=wallet.address, to_address=data["to_address"], value=data["value"], gas=data["gas"], gas_price=data["gas_price"])
        db.session.add(transaction)
        db.session.commit()
        return jsonify({"message": "Transaction created successfully"})
    return jsonify({"message": "Wallet not found"}), 404

app.register_blueprint(user_routes)
app.register_blueprint(wallet_routes)
app.register_blueprint(transaction_routes)
