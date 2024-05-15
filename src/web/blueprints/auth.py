from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400

    user = User.query.filter_by(username=username).first()

    if not user or user.password != password:
        return jsonify({'error': 'Invalid username or password'}), 401

    access_token = create_access_token(identity=user.id)

    return jsonify({'access_token': access_token}), 200

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt_identity()
    revoked_token = RevokedToken.query.filter_by(jti=jti).first()

    if revoked_token is None:
        revoked_token = RevokedToken(jti=jti)
        db.session.add(revoked_token)

    db.session.commit()

    return jsonify({'message': 'Access token revoked'}), 200

@auth_bp.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')

    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400

    user = User(username=username, password=password)

    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'User created'}), 201
