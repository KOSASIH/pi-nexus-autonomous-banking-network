from flask import request, jsonify
from flask_login import login_required

from . import api_bp
from ..models import User
from ..schemas import UserSchema

# Initialize the user schema
user_schema = UserSchema()
users_schema = UserSchema(many=True)

@api_bp.route('/users', methods=['GET'])
@login_required
def get_users():
    """
    Get all users.
    """
    users = User.query.all()
    return users_schema.jsonify(users)

@api_bp.route('/users/<int:user_id>', methods=['GET'])
@login_required
def get_user(user_id):
    """
    Get a user by ID.
    """
    user = User.query.get_or_404(user_id)
    return user_schema.jsonify(user)

@api_bp.route('/users', methods=['POST'])
@login_required
def create_user():
    """
    Create a new user.
    """
    data = request.get_json()
    user = User.from_dict(data)
    db.session.add(user)
    db.session.commit()
    return user_schema.jsonify(user), 201

@api_bp.route('/users/<int:user_id>', methods=['PUT'])
@login_required
def update_user(user_id):
    """
    Update a user by ID.
    """
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    user.update_from_dict(data)
    db.session.commit()
    return user_schema.jsonify(user)

@api_bp.route('/users/<int:user_id>', methods=['DELETE'])
@login_required
def delete_user(user_id):
    """
    Delete a user by ID.
    """
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return '', 204
