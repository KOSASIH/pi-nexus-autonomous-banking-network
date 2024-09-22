from flask import request, jsonify, current_app
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required
from flask_bcrypt import Bcrypt
from .models import User
from .schemas import UserSchema

bcrypt = Bcrypt()

@auth_blueprint.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    return jsonify(error="Invalid credentials"), 401

@auth_blueprint.route("/register", methods=["POST"])
def register():
    username = request.json.get("username")
    password = request.json.get("password")
    user = User(username=username)
    user.password = bcrypt.generate_password_hash(password).decode("utf-8")
    db.session.add(user)
    db.session.commit()
    return jsonify(message="User created successfully")

@auth_blueprint.route("/me", methods=["GET"])
@jwt_required
def me():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    return jsonify(UserSchema().dump(user))

@auth_blueprint.route("/logout", methods=["POST"])
@jwt_required
def logout():
    current_app.jwt_manager.blacklist_token(get_jwt_identity())
    return jsonify(message="Logged out successfully")
