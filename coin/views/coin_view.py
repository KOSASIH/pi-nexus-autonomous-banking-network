from flask import Blueprint, request, jsonify
from .models import Coin

coin_view = Blueprint('coin_view', __name__)

@coin_view.route('/coins', methods=['GET'])
def get_coins():
    coins = Coin.query.all()
    return jsonify([coin.to_dict() for coin in coins])

@coin_view.route('/coins/<int:coin_id>', methods=['GET'])
def get_coin(coin_id):
    coin = Coin.query.get(coin_id)
    if coin:
        return jsonify(coin.to_dict())
    return jsonify({'error': 'Coin not found'}), 404
