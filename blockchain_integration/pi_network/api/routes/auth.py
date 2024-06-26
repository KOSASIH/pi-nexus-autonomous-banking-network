from flask import Blueprint, request, jsonify
from api.models import User
from api.schemas import UserSchema

auth_blueprint = Blueprint("auth", __name__)

@auth_blueprint.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    user = User(**data)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created successfully"})

@auth_blueprint.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data["username"]).first()
    if user and user.password == data["password"]:
        return jsonify({"token": "some_token"})
    return jsonify({"error": "Invalid credentials"}), 401
