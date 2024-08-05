import os
import json
import jwt
from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///token_api.db"
app.config["JWT_SECRET_KEY"] = "super-secret-key"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 3600

db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt_manager = JWTManager(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class UserSchema(ma.Schema):
    class Meta:
        fields = ("id", "username")

user_schema = UserSchema()
users_schema = UserSchema(many=True)

@app.route("/register", methods=["POST"])
def register():
    username = request.json["username"]
    password = request.json["password"]
    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=username)
        return jsonify({"access_token": access_token}), 200
    return jsonify({"message": "Invalid credentials"}), 401

@app.route("/protected", methods=["GET"])
@jwt_required
def protected():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    return jsonify({"message": f"Hello, {user.username}!"}), 200

@app.route("/users", methods=["GET"])
@jwt_required
def get_users():
    users = User.query.all()
    return users_schema.jsonify(users), 200

@app.route("/users/<id>", methods=["GET"])
@jwt_required
def get_user(id):
    user = User.query.get(id)
    if user:
        return user_schema.jsonify(user), 200
    return jsonify({"message": "User not found"}), 404

@app.route("/users/<id>", methods=["PUT"])
@jwt_required
def update_user(id):
    user = User.query.get(id)
    if user:
        username = request.json["username"]
        password = request.json["password"]
        user.username = username
        user.set_password(password)
        db.session.commit()
        return jsonify({"message": "User updated successfully"}), 200
    return jsonify({"message": "User not found"}), 404

@app.route("/users/<id>", methods=["DELETE"])
@jwt_required
def delete_user(id):
    user = User.query.get(id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": "User deleted successfully"}), 200
    return jsonify({"message": "User not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
