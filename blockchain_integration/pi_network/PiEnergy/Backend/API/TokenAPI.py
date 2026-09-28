import os
import secrets
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# --- SECURITY FIX 1: JANGAN HARDCODE SECRET ---
# Ambil dari ENV, wajib set di production
# Di local: export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not JWT_SECRET_KEY:
    # Kalau di production gak di-set, langsung matiin app biar gak deploy dengan kunci lemah
    if os.getenv("FLASK_ENV") == "production":
        raise RuntimeError("FATAL: JWT_SECRET_KEY env var belum di-set!")
    # Kalau di dev, generate random sekali jalan + warning
    JWT_SECRET_KEY = secrets.token_hex(32)
    print(f"WARNING: Using temporary random secret for DEV ONLY: {JWT_SECRET_KEY[:10]}... Set JWT_SECRET_KEY env var!")

if len(JWT_SECRET_KEY) < 32:
    raise RuntimeError("FATAL: JWT_SECRET_KEY terlalu pendek! Minimal 32 karakter, pakai 64 hex char.")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///token_api.db"
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(seconds=3600)

db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt_manager = JWTManager(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    # OPTIONAL: tambah role buat admin check
    role = db.Column(db.String(20), default="user")

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class UserSchema(ma.Schema):
    class Meta:
        fields = ("id", "username", "role")

user_schema = UserSchema()
users_schema = UserSchema(many=True)

@app.route("/register", methods=["POST"])
def register():
    username = request.json.get("username")
    password = request.json.get("password")
    if not username or not password:
        return jsonify({"message": "username & password required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"message": "User already exists"}), 409
    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        # FIX 2: identity sekarang object, bukan cuma string, biar bisa cek role
        access_token = create_access_token(identity={"id": user.id, "username": user.username, "role": user.role})
        return jsonify({"access_token": access_token}), 200
    return jsonify({"message": "Invalid credentials"}), 401

@app.route("/protected", methods=["GET"])
@jwt_required()  # FIX 3: dulu kamu tulis @jwt_required tanpa (), ini gak ngecek token!
def protected():
    current = get_jwt_identity()
    return jsonify({"message": f"Hello, {current['username']}!"}), 200

@app.route("/users", methods=["GET"])
@jwt_required()
def get_users():
    # FIX 4: Cek authorization - cuma admin boleh list semua user
    current = get_jwt_identity()
    if current.get("role") != "admin":
        return jsonify({"message": "Forbidden: admin only"}), 403
    users = User.query.all()
    return users_schema.jsonify(users), 200

@app.route("/users/<int:id>", methods=["GET"])
@jwt_required()
def get_user(id):
    current = get_jwt_identity()
    # User boleh lihat dirinya sendiri, admin boleh lihat semua
    if current["id"] != id and current.get("role") != "admin":
        return jsonify({"message": "Forbidden"}), 403
    user = User.query.get(id)
    if user:
        return user_schema.jsonify(user), 200
    return jsonify({"message": "User not found"}), 404

@app.route("/users/<int:id>", methods=["PUT"])
@jwt_required()
def update_user(id):
    current = get_jwt_identity()
    if current["id"] != id and current.get("role") != "admin":
        return jsonify({"message": "Forbidden"}), 403
    user = User.query.get(id)
    if user:
        username = request.json.get("username")
        password = request.json.get("password")
        if username:
            user.username = username
        if password:
            user.set_password(password)
        db.session.commit()
        return jsonify({"message": "User updated successfully"}), 200
    return jsonify({"message": "User not found"}), 404

@app.route("/users/<int:id>", methods=["DELETE"])
@jwt_required()
def delete_user(id):
    current = get_jwt_identity()
    if current["id"] != id and current.get("role") != "admin":
        return jsonify({"message": "Forbidden"}), 403
    user = User.query.get(id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": "User deleted successfully"}), 200
    return jsonify({"message": "User not found"}), 404

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False) # Jangan debug=True di prod
