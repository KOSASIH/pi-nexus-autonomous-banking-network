# payment_gateway/routes.py
from flask import Blueprint, request, jsonify
from. import app, db, jwt, bcrypt
from.models import User, PaymentMethod, Transaction
from.schemas import UserSchema, PaymentMethodSchema, TransactionSchema
from. import login_manager
import stripe

user_routes = Blueprint("user_routes", __name__)
payment_routes = Blueprint("payment_routes", __name__)
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

@payment_routes.route("/payment-methods", methods=["GET"])
@login_required
def get_payment_methods():
    payment_methods = PaymentMethod.query.all()
    payment_method_schema = PaymentMethodSchema(many=True)
    return jsonify(payment_method_schema.dump(payment_methods))

@payment_routes.route("/payment-methods", methods=["POST"])
@login_required
def create_payment_method():
    data = request.get_json()
    payment_method = PaymentMethod(user_id=current_user.id, payment_method_type=data["payment_method_type"], payment_method_token=data["payment_method_token"])
    db.session.add(payment_method)
    db.session.commit()
    return jsonify({"message": "Payment method created successfully"})

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
    payment_method = PaymentMethod.query.get(data["payment_method_id"])
    if payment_method:
        transaction = Transaction(user_id=current_user.id, payment_method_id=payment_method.id, amount=data["amount"], currency=data["currency"])
        db.session.add(transaction)
        db.session.commit()
        stripe.Charge.create(
            amount=int(data["amount"] * 100),
            currency=data["currency"],
            source=payment_method.payment_method_token,
            description="Test transaction"
        )
        return jsonify({"message": "Transaction created successfully"})
    return jsonify({"message": "Payment method not found"}), 404

app.register_blueprint(user_routes)
app.register_blueprint(payment_routes)
app.register_blueprint(transaction_routes)
