# blockchain_integration/__init__.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_bcrypt import Bcrypt
from web3 import Web3, HTTPProvider

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///blockchain_integration.db"
app.config["JWT_SECRET_KEY"] = "super-secret-key"
app.config["BCRYPT_LOG_ROUNDS"] = 12
app.config["BLOCKCHAIN_NETWORK"] = "mainnet"
app.config["BLOCKCHAIN_NODE_URL"] = "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"

db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
w3 = Web3(HTTPProvider(app.config["BLOCKCHAIN_NODE_URL"]))

from.models import User, Wallet, Transaction
from.schemas import UserSchema, WalletSchema, TransactionSchema
from.routes import user_routes, wallet_routes, transaction_routes

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
