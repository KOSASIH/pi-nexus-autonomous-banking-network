from flask import request, jsonify
from models import Coin, Wallet, Transaction, User
from utils import get_coin_info

# Get all coins
def get_coins():
    coins = Coin.query.all()
    return jsonify([coin.to_dict() for coin in coins])

# Get coin by symbol
def get_coin(symbol):
    coin = Coin.query.filter_by(symbol=symbol).first()
    if coin:
        return jsonify(coin.to_dict())
    else:
        return jsonify({'error': 'Coin not found'}), 404

# Create or update coin
def create_or_update_coin():
    data = request.get_json()
    coin = Coin.query.filter_by(symbol=data['symbol']).first()
    if coin:
        coin.name = data['name']
        coin.price = data['price']
        coin.market_cap = data['market_cap']
        coin.volume = data['volume']
    else:
        coin = Coin(symbol=data['symbol'], name=data['name'], price=data['price'], market_cap=data['market_cap'], volume=data['volume'])
    db.session.add(coin)
    db.session.commit()
    return jsonify(coin.to_dict())

# Delete coin
def delete_coin(symbol):
    coin = Coin.query.filter_by(symbol=symbol).first()
    if coin:
        db.session.delete(coin)
        db.session.commit()
        return jsonify({'message': 'Coin deleted'})
    else:
        return jsonify({'error': 'Coin not found'}), 404

# Get user's wallet
def get_wallet(user_id):
    user = User.query.get(user_id)
    if user:
        return jsonify([wallet.to_dict() for wallet in user.wallets])
    else:
        return jsonify({'error': 'User not found'}), 404

# Create or update user's wallet
def create_or_update_wallet():
    data = request.get_json()
    user = User.query.get(data['user_id'])
    coin = Coin.query.filter_by(symbol=data['coin_symbol']).first()
    if user and coin:
        wallet = Wallet.query.filter_by(user_id=user.id, coin_id=coin.id).first()
        if wallet:
            wallet.balance = data['balance']
        else:
            wallet = Wallet(user_id=user.id, coin_id=coin.id, balance=data['balance'])
        db.session.add(wallet)
        db.session.commit()
        return jsonify(wallet.to_dict())
    else:
        return jsonify({'error': 'User or Coin not found'}), 404

# Delete user's wallet
def delete_wallet(wallet_id):
    wallet = Wallet.query.get(wallet_id)
    if wallet:
        db.session.delete(wallet)
        db.session.commit()
        return jsonify({'message': 'Wallet deleted'})
    else:
        return jsonify({'error': 'Wallet not found'}), 404

# Get user's transaction history
def get_transactions(wallet_id):
    transactions = Transaction.query.filter_by(wallet_id=wallet_id).all()
    if transactions:
        return jsonify([transaction.to_dict() for transaction in transactions])
    else:
        return jsonify({'error': 'No transactions found'}), 404

# Create new transaction
def create_transaction():
    data = request.get_json()
    wallet = Wallet.query.get(data['wallet_id'])
    if wallet:
        transaction = Transaction(wallet_id=wallet.id, amount=data['amount'], type=data['type'])
        db.session.add(transaction)
        db.session.commit()
        return jsonify(transaction.to_dict())
    else:
        return jsonify({'error': 'Wallet not found'}), 404
