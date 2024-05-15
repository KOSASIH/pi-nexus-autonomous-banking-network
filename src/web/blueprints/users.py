from flask import Blueprint, jsonify, request
from flask_sqlalchemy import sqlalchemy

from web.models import User

users_bp = Blueprint("users", __name__, url_prefix="/api/users")


@users_bp.route("", methods=["GET"])
@jwt_required()
def get_users():
    users = User.query.all()

    return jsonify([user.to_dict() for user in users]), 200


@users_bp.route("/<int:user_id>", methods=["GET"])
@jwt_required()
def get_user(user_id):
    user = User.query.get_or_404(user_id)

    return jsonify(user.to_dict()), 200


@users_bp.route("", methods=["POST"])
def create_user():
    username = request.json.get("username")
    password = request.json.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    user = User(username=username, password=password)

    db.session.add(user)
    db.session.commit()

    return jsonify(user.to_dict()), 201


@users_bp.route("/<int:user_id>", methods=["PUT"])
@jwt_required()
def update_user(user_id):
    user = User.query.get_or_404(user_id)

    username = request.json.get("username")
    password = request.json.get("password")

    if username:
        user.username = username

    if password:
        user.password = password

    db.session.commit()

    return jsonify(user.to_dict()), 200


@users_bp.route("/<int:user_id>", methods=["DELETE"])
@jwt_required()
def delete_user(user_id):
    user = User.query.get_or_404(user_id)

    db.session.delete(user)
    db.session.commit()

    return jsonify({"message": "User deleted"}), 200
