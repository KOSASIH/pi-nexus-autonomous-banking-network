import os
import json
import hashlib
import time
from collections import namedtuple
from typing import List, Optional, Dict
import requests
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///lab.db"
app.config["JWT_SECRET_KEY"] = "super-secret-key"
db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt = JWTManager(app)

# Define named tuples for data structures
User = namedtuple('User', 'id, username, password, email')
Account = namedtuple('Account', 'id, user_id, balance, account_type')
Transaction = namedtuple('Transaction', 'id, sender_id, receiver_id, amount, timestamp')

# Define database models
class UserModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)

class AccountModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_model.id'), nullable=False)
    balance = db.Column(db.Float, default=0.0, nullable=False)
    account_type = db.Column(db.String(100), nullable=False)

class TransactionModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user_model.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user_model.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)

# Define schema for database models
class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = UserModel
        load_instance = True

class AccountSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = AccountModel
        load_instance = True

class TransactionSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = TransactionModel
        load_instance = True

# Define API endpoints
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    user = UserModel(username=data['username'], password=data['password'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User created successfully'})

@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    user = UserModel.query.filter_by(username=data['username'], password=data['password']).first()
    if user:
        access_token = create_access_token(identity=user.id)
        return jsonify({'access_token': access_token})
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/accounts', methods=['GET'])
@jwt_required
def get_accounts():
    user_id = get_jwt_identity()
    accounts = AccountModel.query.filter_by(user_id=user_id).all()
    return jsonify(AccountSchema(many=True).dump(accounts))

@app.route('/transactions', methods=['GET'])
@jwt_required
def get_transactions():
    user_id = get_jwt_identity()
    transactions = TransactionModel.query.filter_by(sender_id=user_id).all()
    return jsonify(TransactionSchema(many=True).dump(transactions))

@app.route('/transfer', methods=['POST'])
@jwt_required
def transfer_funds():
    data = request.get_json()
    sender_id = get_jwt_identity()
    receiver_id = UserModel.query.filter_by(username=data['receiver_username']).first().id
    amount = data['amount']
    transaction = TransactionModel(sender_id=sender_id, receiver_id=receiver_id, amount=amount)
    db.session.add(transaction)
    db.session.commit()
    return jsonify({'message': 'Transaction successful'})

@app.route('/balance', methods=['GET'])
@jwt_required
def get_balance():
    user_id = get_jwt_identity()
    account = AccountModel.query.filter_by(user_id=user_id).first()
    return jsonify({'balance': account.balance})

if __name__ == '__main__':
    app.run(debug=True)
