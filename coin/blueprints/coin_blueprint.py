from flask import Blueprint, jsonify
from services.coin_service import CoinService

coin_blueprint = Blueprint('coin_blueprint', __name__)

@coin_blueprint.route('/coins', methods=['GET'])
def get_coins():
    coin_service = CoinService()
    coins = coin_service.get_coins()
    return jsonify([coin.serialize for coin in coins])

@coin_blueprint.route('/coins', methods=['POST'])
def create_coin():
    # Parse request data and create a coin
    pass
