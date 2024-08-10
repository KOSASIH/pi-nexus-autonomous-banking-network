from flask import Blueprint, request, jsonify
from models import Transaction
from services import TransactionService
from utils import hash_data

transaction_blueprint = Blueprint('transaction', __name__)

@transaction_blueprint.route('/transaction', methods=['GET'])
def get_transactions():
    transaction_service = TransactionService()
    transactions = transaction_service.get_transactions()
    return jsonify([transaction.__dict__ for transaction in transactions])

@transaction_blueprint.route('/transaction', methods=['POST'])
def create_transaction():
    data = request.get_json()
    sender = data['sender']
    recipient = data['recipient']
    amount = data['amount']
    transaction = Transaction(sender, recipient, amount)
    transaction_service = TransactionService()
    transaction_service.add_transaction(transaction)
    return jsonify(transaction.__dict__), 201

@transaction_blueprint.route('/transaction/<string:transaction_id>', methods=['GET'])
def get_transaction(transaction_id):
    transaction_service = TransactionService()
    transaction = transaction_service.get_transaction(transaction_id)
    if transaction is None:
        return jsonify({'error': 'Transaction not found'}), 404
    return jsonify(transaction.__dict__)

@transaction_blueprint.route('/transaction/<string:transaction_id>', methods=['PUT'])
def update_transaction(transaction_id):
    data = request.get_json()
    sender = data['sender']
    recipient = data['recipient']
    amount = data['amount']
    transaction_service = TransactionService()
    transaction = transaction_service.get_transaction(transaction_id)
    if transaction is None:
        return jsonify({'error': 'Transaction not found'}), 404
    transaction.sender = sender
    transaction.recipient = recipient
    transaction.amount = amount
    transaction_service.update_transaction(transaction)
    return jsonify(transaction.__dict__)

@transaction_blueprint.route('/transaction/<string:transaction_id>', methods=['DELETE'])
def delete_transaction(transaction_id):
    transaction_service = TransactionService()
    transaction = transaction_service.get_transaction(transaction_id)
    if transaction is None:
        return jsonify({'error': 'Transaction not found'}), 404
    transaction_service.delete_transaction(transaction)
    return jsonify({'message': 'Transaction deleted'})

@transaction_blueprint.route('/transaction/validate', methods=['POST'])
def validate_transaction():
    data = request.get_json()
    sender = data['sender']
    recipient = data['recipient']
    amount = data['amount']
    transaction = Transaction(sender, recipient, amount)
    transaction_service = TransactionService()
    if not transaction_service.validate_transaction(transaction):
        return jsonify({'error': 'Invalid transaction'}), 400
    return jsonify({'message': 'Transaction is valid'})
