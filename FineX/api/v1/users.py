# users.py

from flask import Blueprint, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from marshmallow import Schema, fields, ValidationError
import jwt
import datetime

# Initialize the database
db = SQLAlchemy()

# Create a blueprint for users
users_bp = Blueprint('users', __name__)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# User schema for validation
class UserSchema(Schema):
    username = fields.Str(required=True)
    email = fields.Email(required=True)
    password = fields.Str(required=True)

user_schema = UserSchema()
users_schema = UserSchema(many=True)

# Secret key for JWT
SECRET_KEY = 'your_secret_key_here'

# Route to register a new user
@users_bp.route('/users/register', methods=['POST'])
def register_user():
    try:
        # Validate and deserialize input
        data = user_schema.load(request.json)
        hashed_password = generate_password_hash(data['password'], method='sha256')
        new_user = User(username=data['username'], email=data['email'], password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return user_schema.jsonify(new_user), 201
    except ValidationError as err:
        return jsonify(err.messages), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to authenticate a user
@users_bp.route('/users/login', methods=['POST'])
def login_user():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if user and check_password_hash(user.password, data['password']):
        token = jwt.encode({'user_id': user.id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, SECRET_KEY)
        return jsonify({'token': token}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

# Route to get user profile
@users_bp.route('/users/profile', methods=['GET'])
def get_user_profile():
    token = request.headers.get('Authorization').split()[1]
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user = User.query.get(decoded['user_id'])
        return user_schema.jsonify(user), 200
    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token'}), 401

# Route to update user profile
@users_bp.route('/users/profile', methods=['PUT'])
def update_user_profile():
    token = request.headers.get('Authorization').split()[1]
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user = User.query.get(decoded['user_id'])
        data = request.json
        if 'username' in data:
            user.username = data['username']
        if 'email' in data:
            user.email = data['email']
        if 'password' in data:
            user.password = generate_password_hash(data['password'], method='sha256')
        db.session.commit()
        return user_schema.jsonify(user), 200
    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token'}), 401

# Route to delete a user
@users_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User  deleted successfully."}), 204

# Register the blueprint in the main application
def init_app(app):
    app.register_blueprint(users_bp, url_prefix='/api/v1')
