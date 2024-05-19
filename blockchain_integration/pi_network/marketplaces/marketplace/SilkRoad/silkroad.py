import os
import json
import hashlib
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length
from cryptography.fernet import Fernet
from silkroad.utils import generate_token, verify_token

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///silkroad.db"
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default="customer")

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500), nullable=False)
    price = db.Column(db.Float, nullable=False)
    stock = db.Column(db.Integer, nullable=False)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"))
    quantity = db.Column(db.Integer, nullable=False)
    total = db.Column(db.Float, nullable=False)

class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8)])
    submit = SubmitField("Register")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == hashlib.sha256(form.password.data.encode()).hexdigest():
            login_user(user)
            return redirect(url_for("dashboard"))
    return render_template("login.html", form=form)

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data, password=hashlib.sha256(form.password.data.encode()).hexdigest())
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html", form=form)

@app.route("/dashboard")
@login_required
def dashboard():
    products = Product.query.all()
    return render_template("dashboard.html", products=products)

@app.route("/product/<int:product_id>")
@login_required
def product(product_id):
    product = Product.query.get_or_404(product_id)
    return render_template("product.html", product=product)

@app.route("/order", methods=["POST"])
@login_required
def order():
    product_id = request.form["product_id"]
    quantity = int(request.form["quantity"])
    product = Product.query.get_or_404(product_id)
    if product.stock >= quantity:
        order = Order(user_id=current_user.id, product_id=product_id, quantity=quantity, total=product.price * quantity)
        db.session.add(order)
        db.session.commit()
        return jsonify({"message": "Order placed successfully"})
    return jsonify({"message": "Insufficient stock"}), 400

@app.route("/orders")
@login_required
def orders():
    orders = Order.query.filter_by(user_id=current_user.id).all()
    return render_template("orders.html", orders=orders)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/api/token", methods=["POST"])
def generate_token():
    username = request.form["username"]
    password = request.form["password"]
    user = User.query.filter_by(username=username).first()
    if user and user.password == hashlib.sha256(password.encode()).hexdigest():
        token = generate_token(user.id)
        return jsonify({"token": token})
    return jsonify({"message": "Invalid username or password"}), 400

@app.route("/api/orders", methods=["GET"])
def get_orders():
    token = request.args["token"]
    user_id = validate_token(token)
    if user_id:
        orders = Order.query.filter_by(user_id=user_id).all()
        return jsonify({"orders": [order.serialize for order in orders]})
    return jsonify({"message": "Invalid token"}), 400

@app.route("/api/order", methods=["POST"])
def place_order():
    token = request.form["token"]
    user_id = validate_token(token)
    product_id = int(request.form["product_id"])
    quantity = int(request.form["quantity"])
    if user_id:
        product = Product.query.get_or_404(product_id)
        if product.stock >= quantity:
            order = Order(user_id=user_id, product_id=product_id, quantity=quantity, total=product.price * quantity)
            db.session.add(order)
            db.session.commit()
            return jsonify({"message": "Order placed successfully"})
        return jsonify({"message": "Insufficient stock"}), 400
    return jsonify({"message": "Invalid token"}), 400

@app.route("/api/order/<int:order_id>", methods=["DELETE"])
def delete_order(order_id):
    token = request.args["token"]
    user_id = validate_token(token)
    if user_id:
        order = Order.query.get_or_404(order_id)
        if order.user_id == user_id:
            db.session.delete(order)
            db.session.commit()
            return jsonify({"message": "Order deleted successfully"})
        return jsonify({"message": "You can only delete your own orders"}), 400
    return jsonify({"message": "Invalid token"}), 400

if __name__ == "__main__":
    app.run(debug=True)```

This code includes the following features:

1. User registration and login
2. Product browsing and ordering
3. Order management
4. Token-based API for order management

Please note that this is a simplified example and should not be used as-is in a production environment. Additional features, security measures, and error handling should be implemented.```
