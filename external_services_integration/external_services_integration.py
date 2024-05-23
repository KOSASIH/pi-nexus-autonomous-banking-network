import os
import requests
import json
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from cryptography.fernet import Fernet

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
db = SQLAlchemy(app)

# Load encryption key from environment variable
encryption_key = os.environ.get("ENCRYPTION_KEY")
fernet = Fernet(encryption_key)

# PayPal API credentials
paypal_client_id = os.environ.get("PAYPAL_CLIENT_ID")
paypal_client_secret = os.environ.get("PAYPAL_CLIENT_SECRET")

# CardX API credentials
cardx_api_key = os.environ.get("CARDX_API_KEY")
cardx_api_secret = os.environ.get("CARDX_API_SECRET")

# Database model for user accounts
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    paypal_linked = db.Column(db.Boolean, default=False)
    cardx_linked = db.Column(db.Boolean, default=False)
    paypal_token = db.Column(db.String(255), nullable=True)
    cardx_token = db.Column(db.String(255), nullable=True)

@app.route("/link_paypal", methods=["POST"])
def link_paypal():
    username = request.json["username"]
    password = request.json["password"]
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        # Authenticate with PayPal API
        auth_response = requests.post(
            "https://api.paypal.com/v1/oauth2/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": paypal_client_id,
                "client_secret": paypal_client_secret
            }
        )
        if auth_response.status_code == 200:
            paypal_token = auth_response.json()["access_token"]
            user.paypal_linked = True
            user.paypal_token = paypal_token
            db.session.commit()
            return jsonify({"message": "PayPal account linked successfully"})
        else:
            return jsonify({"error": "Failed to authenticate with PayPal"}), 401
    else:
        return jsonify({"error": "Invalid username or password"}), 401

@app.route("/link_cardx", methods=["POST"])
def link_cardx():
    username = request.json["username"]
    password = request.json["password"]
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        # Authenticate with CardX API
        auth_response = requests.post(
            "https://api.cardx.com/v1/authenticate",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": cardx_api_key,
                "api_secret": cardx_api_secret
            }
        )
        if auth_response.status_code == 200:
            cardx_token = auth_response.json()["token"]
            user.cardx_linked = True
            user.cardx_token = cardx_token
            db.session.commit()
            return jsonify({"message": "CardX account linked successfully"})
        else:
            return jsonify({"error": "Failed to authenticate with CardX"}), 401
    else:
        return jsonify({"error": "Invalid username or password"}), 401

@app.route("/make_payment", methods=["POST"])
def make_payment():
    username = request.json["username"]
    password = request.json["password"]
    payment_amount = request.json["amount"]
    payment_method = request.json["method"]
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        if payment_method == "paypal":
            if user.paypal_linked:
                # Use PayPal API to make payment
                payment_response = requests.post(
                    "https://api.paypal.com/v1/payments/payment",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {user.paypal_token}"},
                    json={
                        "intent": "sale",
                        "payer": {"payment_method": "paypal"},
                        "transactions": [{"amount": {"currency": "USD", "total": payment_amount}}]
                    }
                )
                if payment_response.status_code == 200:
                    return jsonify({"message": "Payment made successfully"})
                else:
                    return jsonify({"error": "Failed to make payment with PayPal"}), 500
            else:
                return jsonify({"error": "PayPal account not linked"}), 401
elif payment_method == "cardx":
            if user.cardx_linked:
                # Use CardX API to make payment
                payment_response = requests.post(
                    "https://api.cardx.com/v1/payments",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {user.cardx_token}"},
                    json={
                        "amount": payment_amount,
                        "card_number": request.json["card_number"],
                        "expiration_date": request.json["expiration_date"],
                        "cvv": request.json["cvv"]
                    }
                )
                if payment_response.status_code == 200:
                    return jsonify({"message": "Payment made successfully"})
                else:
                    return jsonify({"error": "Failed to make payment with CardX"}), 500
            else:
                return jsonify({"error": "CardX account not linked"}), 401
        else:
            return jsonify({"error": "Invalid payment method"}), 400
    else:
        return jsonify({"error": "Invalid username or password"}), 401

if __name__ == "__main__":
    app.run(debug=True)
