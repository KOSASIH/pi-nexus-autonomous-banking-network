from flask import Blueprint, request, jsonify
from.models import Wallet

wallet_view = Blueprint('wallet_view', __name__)

@wallet_view.route('/wallets', methods=['GET'])
def get_wallets():
    wallets = Wallet.query.all()
    return jsonify([wallet.to_dict() for wallet in wallets])

@wallet_view.route('/wallets/<int:wallet_id>', methods=['GET'])
def get_wallet(wallet_id):
    wallet = Wallet.query.get(wallet_id)
    if wallet:
        return jsonify(wallet.to_dict())
    return jsonify({'error': 'Wallet not found'}), 404
