# user_management/routes.py
from flask import Blueprint, request, jsonify
from . import app, db, jwt, bcrypt
from .models import User
from .schemas import UserSchema
from . import login_manager

user_routes = Blueprint("user_routes", __name__)
auth_routes = Blueprint("auth_routes", __name__)

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

@auth_routes.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data["username"]).first()
    if user and user.check_password(data["password"]):
        access_token = create_access_token(identity=user.username)
        login_user(user)
        return jsonify({"access_token": access_token})
    return jsonify({"message": "Invalid credentials"}), 401

@auth_routes.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

app.register_blueprint(user_routes)
app.register_blueprint(auth_routes)
