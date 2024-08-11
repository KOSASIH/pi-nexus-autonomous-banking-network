# main.py

import os
import json
import logging
from flask import Flask, request, jsonify, g
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_cors import CORS
from app.models import User, Transaction, Account, Node, Block, Blockchain
from app.services import AuthService, PaymentService, AnalyticsService, BlockchainService
from app.utils import constants, helpers
from app.blockchain import BlockchainUtils

app = Flask(__name__)
app.config.from_object(constants.Config)

db = SQLAlchemy(app)
ma = Marshmallow(app)
jwt = JWTManager(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.before_request
def before_request():
    g.user = None
    if 'Authorization' in request.headers:
        token = request.headers['Authorization'].split()[1]
        g.user = get_jwt_identity()

@app.route('/api/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    auth_service = AuthService()
    response = auth_service.login(username, password)
    return jsonify(response), 200 if response.get('access_token') else 401

@app.route('/api/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')
    email = request.json.get('email')
    auth_service = AuthService()
    response = auth_service.register(username, password, email)
    return jsonify(response), 201

@app.route('/api/users', methods=['GET'])
@jwt_required
def get_users():
    user_service = UserService()
    users = user_service.get_users()
    return jsonify([user.to_dict() for user in users]), 200

@app.route('/api/transactions', methods=['GET'])
@jwt_required
def get_transactions():
    transaction_service = TransactionService()
    transactions = transaction_service.get_transactions()
    return jsonify([transaction.to_dict() for transaction in transactions]), 200

@app.route('/api/accounts', methods=['GET'])
@jwt_required
def get_accounts():
    account_service = AccountService()
    accounts = account_service.get_accounts()
    return jsonify([account.to_dict() for account in accounts]), 200

@app.route('/api/make_payment', methods=['POST'])
@jwt_required
def make_payment():
    user_id = g.user
    account_id = request.json.get('account_id')
    amount = request.json.get('amount')
    payment_service = PaymentService()
    response = payment_service.make_payment(user_id, account_id, amount)
    return jsonify(response), 200 if response.get('message') == 'Payment successful' else 400

@app.route('/api/get_payment_history', methods=['GET'])
@jwt_required
def get_payment_history():
    user_id = g.user
    payment_service = PaymentService()
    transactions = payment_service.get_payment_history(user_id)
    return jsonify(transactions), 200

@app.route('/api/analytics', methods=['GET'])
@jwt_required
def get_analytics():
    analytics_service = AnalyticsService()
    top_spenders = analytics_service.get_top_spenders()
    transaction_volume = analytics_service.get_transaction_volume()
    return jsonify({'top_spenders': top_spenders, 'transaction_volume': transaction_volume}), 200

@app.route('/api/blockchain', methods=['GET'])
@jwt_required
def get_blockchain():
    blockchain_service = BlockchainService()
    blockchain = blockchain_service.get_blockchain()
    return jsonify(blockchain), 200

@app.route('/api/nodes', methods=['GET'])
@jwt_required
def get_nodes():
    node_service = NodeService()
    nodes = node_service.get_nodes()
    return jsonify([node.to_dict() for node in nodes]), 200

@app.route('/api/blocks', methods=['GET'])
@jwt_required
def get_blocks():
    block_service = BlockService()
    blocks = block_service.get_blocks()
    return jsonify([block.to_dict() for block in blocks]), 200

@app.route('/api/transactions/pending', methods=['GET'])
@jwt_required
def get_pending_transactions():
    transaction_service = TransactionService()
    transactions = transaction_service.get_pending_transactions()
    return jsonify([transaction.to_dict() for transaction in transactions]), 200

@app.route('/api/transactions/confirmed', methods=['GET'])
@jwt_required
def get_confirmed_transactions():
    transaction_service = TransactionService()
    transactions = transaction_service.get_confirmed_transactions()
    return jsonify([transaction.to_dict() for transaction in transactions]), 200

@app.route('/api/blockchain/sync', methods=['POST'])
@jwt_required
def sync_blockchain():
    blockchain_service = BlockchainService()
    response = blockchain_service.sync_blockchain()
    return jsonify(response), 200

@app.route('/api/blockchain/mine', methods=['POST'])
@jwt_required
def mine_block():
    blockchain_service = BlockchainService()
    response = blockchain_service.mine_block()
    return jsonify(response), 200

@app.route('/api/blockchain/validate', methods=['POST'])
@jwt_required
def validate_blockchain():
    blockchain_service = BlockchainService()
    response = blockchain_service.validate_blockchain()
    return jsonify(response), 200

@app.route('/api/nodes/register', methods=['POST'])
@jwt_required
def register_node():
    node_service = NodeService()
    node_data = request.json
    response = node_service.register_node(node_data)
    return jsonify(response), 201

@app.route('/api/nodes/unregister', methods=['POST'])
@jwt_required
def unregister_node():
    node_service = NodeService()
    node_id = request.json.get('node_id')
    response = node_service.unregister_node(node_id)
    return jsonify(response), 200

@app.route('/api/blocks/get', methods=['GET'])
@jwt_required
def get_block():
    block_service = BlockService()
    block_hash = request.args.get('block_hash')
    block = block_service.get_block(block_hash)
    return jsonify(block.to_dict()), 200

@app.route('/api/transactions/get', methods=['GET'])
@jwt_required
def get_transaction():
    transaction_service = TransactionService()
    transaction_id = request.args.get('transaction_id')
    transaction = transaction_service.get_transaction(transaction_id)
    return jsonify(transaction.to_dict()), 200

@app.route('/api/wallet', methods=['GET'])
@jwt_required
def get_wallet():
    wallet_service = WalletService()
    user_id = g.user
    wallet = wallet_service.get_wallet(user_id)
    return jsonify(wallet.to_dict()), 200

@app.route('/api/wallet/transactions', methods=['GET'])
@jwt_required
def get_wallet_transactions():
    wallet_service = WalletService()
    user_id = g.user
    transactions = wallet_service.get_wallet_transactions(user_id)
    return jsonify([transaction.to_dict() for transaction in transactions]), 200

@app.route('/api/wallet/balance', methods=['GET'])
@jwt_required
def get_wallet_balance():
    wallet_service = WalletService()
    user_id = g.user
    balance = wallet_service.get_wallet_balance(user_id)
    return jsonify({'balance': balance}), 200

if __name__ == '__main__':
    app.run(debug=True)
