import os
import json
import logging
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_restful import Api, Resource
from.models import Coin, Wallet, Transaction, User
from.views import coin_view, wallet_view, transaction_view, user_view
from.controllers import coin_controller, wallet_controller, transaction_controller, user_controller
from.utils import coin_utils, wallet_utils, transaction_utils, user_utils
from.ai import predict_model
from.blockchain import blockchain
from.smart_contract import smart_contract

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'qlite:///pi-nexus.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'uper-secret-key'
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'

# Initialize database
db = SQLAlchemy(app)
ma = Marshmallow(app)

# Initialize JWT
jwt = JWTManager(app)

# Initialize API
api = Api(app)

# Define routes
class CoinResource(Resource):
    @jwt_required
    def get(self, coin_id):
        coin = coin_controller.get_coin(coin_id)
        if coin:
            return jsonify(coin.to_dict())
        abort(404)

    @jwt_required
    def put(self, coin_id):
        data = request.get_json()
        coin = coin_controller.update_coin(coin_id, data)
        if coin:
            return jsonify(coin.to_dict())
        abort(404)

    @jwt_required
    def delete(self, coin_id):
        coin_controller.delete_coin(coin_id)
        return jsonify({'message': 'Coin deleted'})

class WalletResource(Resource):
    @jwt_required
    def get(self, wallet_id):
        wallet = wallet_controller.get_wallet(wallet_id)
        if wallet:
            return jsonify(wallet.to_dict())
        abort(404)

    @jwt_required
    def put(self, wallet_id):
        data = request.get_json()
        wallet = wallet_controller.update_wallet(wallet_id, data)
        if wallet:
            return jsonify(wallet.to_dict())
        abort(404)

    @jwt_required
    def delete(self, wallet_id):
        wallet_controller.delete_wallet(wallet_id)
        return jsonify({'message': 'Wallet deleted'})

class TransactionResource(Resource):
    @jwt_required
    def get(self, transaction_id):
        transaction = transaction_controller.get_transaction(transaction_id)
        if transaction:
            return jsonify(transaction.to_dict())
        abort(404)

    @jwt_required
    def put(self, transaction_id):
        data = request.get_json()
        transaction = transaction_controller.update_transaction(transaction_id, data)
        if transaction:
            return jsonify(transaction.to_dict())
        abort(404)

    @jwt_required
    def delete(self, transaction_id):
        transaction_controller.delete_transaction(transaction_id)
        return jsonify({'message': 'Transaction deleted'})

class UserResource(Resource):
    @jwt_required
    def get(self, user_id):
        user = user_controller.get_user(user_id)
        if user:
            return jsonify(user.to_dict())
        abort(404)

    @jwt_required
    def put(self, user_id):
        data = request.get_json()
        user = user_controller.update_user(user_id, data)
        if user:
            return jsonify(user.to_dict())
        abort(404)

    @jwt_required
    def delete(self, user_id):
        user_controller.delete_user(user_id)
        return jsonify({'message': 'User deleted'})

class PredictResource(Resource):
    @jwt_required
    def post(self):
        data = request.get_json()
        prediction = predict_model.predict(data)
        return jsonify({'prediction': prediction})

class BlockchainResource(Resource):
    @jwt_required
    def get(self):
        blockchain_data = blockchain.get_blockchain()
        return jsonify(blockchain_data)

    @jwt_required
    def post(self):
        data = request.get_json()
        blockchain.add_block(data)
        return jsonify({'message': 'Block added'})

class SmartContractResource(Resource):
    @jwt_required
    def post(self):
        data = request.get_json()
        result = smart_contract.execute(data['clause_id'])
        return jsonify({'result': result})

api.add_resource(CoinResource, '/coins/<int:coin_id>')
api.add_resource(WalletResource, '/wallets/<int:wallet_id>')
api.add_resource(TransactionResource, '/transactions/<int:transaction_id>')
api.add_resource(UserResource, '/users/<int:user_id>')
api.add_resource(PredictResource, '/predict')
api.add_resource(BlockchainResource, '/blockchain')
api.add_resource(SmartContractResource, '/smart_contract')

# Define error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized'}), 401

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

# Initialize models
db.create_all()

# Initialize controllers
coin_controller.init_app(app)
wallet_controller.init_app(app)
transaction_controller.init_app(app)
user_controller.init_app(app)

# Initialize utils
coin_utils.init_app(app)
wallet_utils.init_app(app)
transaction_utils.init_app(app)
user_utils.init_app(app)

# Initialize AI model
predict_model.init_app(app)

# Initialize blockchain
blockchain.init_app(app)

# Initialize smart contract
smart_contract.init_app(app)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
