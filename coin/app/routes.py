from flask import Blueprint, jsonify
from controllers import get_coins, get_coin, create_or_update_coin, delete_coin, get_wallet, create_or_update_wallet, delete_wallet, get_transactions

coin_blueprint = Blueprint('coin', __name__)

@coin_blueprint.route('/coins', methods=['GET'])
def coins():
    return get_coins()

@coin_blueprint.route('/coins/<string:symbol>', methods=['GET'])
def coin(symbol):
    return get_coin(symbol)

@coin_blueprint.route('/coins', methods=['POST'])
def create_or_update_coin():
    return create_or_update_coin()

@coin_blueprint.route('/coins/<string:symbol>', methods=['DELETE'])
def delete_coin(symbol):
    return delete_coin(symbol)

wallet_blueprint = Blueprint('wallet', __name__)

@wallet_blueprint.route('/wallets/<int:user_id>', methods=['GET'])
def wallet(user_id):
    return get_wallet(user_id)

@wallet_blueprint.route('/wallets', methods=['POST'])
def create_or_update_wallet():
    return create_or_update_wallet()

@wallet_blueprint.route('/wallets/<int:wallet_id>', methods=['DELETE'])
def delete_wallet(wallet_id):
    return delete_wallet(wallet_id)

transaction_blueprint = Blueprint('transaction', __name__)

@transaction_blueprint.route('/transactions/<int:wallet_id>', methods=['GET'])
def transactions(wallet_id):
    return get_transactions(wallet_id)
