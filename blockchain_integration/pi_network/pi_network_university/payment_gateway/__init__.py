# payment_gateway/__init__.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///payment_gateway.db"
app.config["JWT_SECRET_KEY"] = "super-secret-key"
app.config["BCRYPT_LOG_ROUNDS"] = 12
app.config["STRIPE_SECRET_KEY"] = "sk_test_1234567890"
app.config["STRIPE_PUBLIC_KEY"] = "pk_test_1234567890"

db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

from.models import User, PaymentMethod, Transaction
from.schemas import UserSchema, PaymentMethodSchema, TransactionSchema
from.routes import user_routes, payment_routes, transaction_routes

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
