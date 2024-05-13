from flask import Flask, request, jsonify
from controllers.account_controller import deposit, withdraw, transfer
from controllers.transaction_controller import create_transaction, update_transaction, delete_transaction
from services.account_service import AccountService
from services.transaction_service import TransactionService
from views.account_view import render_account_balance, render_transaction_history
from views.transaction_view import render_transaction_list, render_transaction_details

app = Flask(__name__)

account_service = AccountService()
transaction_service = TransactionService()

@app.route('/deposit', methods=['POST'])
def deposit_route():
    account_number = request.json['account_number']
    amount = request.json['amount']
    account_balance = deposit(account_number, amount)
    transaction_history = render_transaction_history(transaction_service.get_transactions(account_number))
    return jsonify({'account_balance': account_balance, 'transaction_history': transaction_history})

@app.route('/withdraw', methods=['POST'])
def withdraw_route():
    account_number = request.json['account_number']
    amount = request.json['amount']
    account_balance = withdraw(account_number, amount)
    transaction_history = render_transaction_history(transaction_service.get_transactions(account_number))
    return jsonify({'account_balance': account_balance, 'transaction_history': transaction_history})

@app.route('/transfer', methods=['POST'])
def transfer_route():
    from_account_number = request.json['from_account_number']
    to_account_number = request.json['to_account_number']
    amount = request.json['amount']
    from_account_balance = transfer(from_account_number, to_account_number, amount)
    to_account_balance = deposit(to_account_number, amount)
    from_transaction_history = render_transaction_history(transaction_service.get_transactions(from_account_number))
    to_transaction_history = render_transaction_history(transaction_service.get_transactions(to_account_number))
    return jsonify({'from_account_balance': from_account_balance, 'to_account_balance': to_account_balance, 'from_transaction_history': from_transaction_history, 'to_transaction_history': to_transaction_history})

@app.route('/transactions', methods=['POST'])
def transactions_route():
    account_number = request.json['account_number']
    transaction = request.json['transaction']
    transaction_list = create_transaction(account_number, transaction)
    transaction_details = render_transaction_details(transaction)
    return jsonify({'transaction_list': transaction_list, 'transaction_details': transaction_details})

@app.route('/transactions/<transaction_id>', methods=['PUT'])
def update_transaction_route(transaction_id):
    account_number = transaction_service.get_transaction(transaction_id)['account_number']
    transaction = request.json['transaction']
    transaction_list = update_transaction(transaction_id, transaction)
    transaction_details = render_transaction_details(transaction)
    return jsonify({'transaction_list': transaction_list, 'transaction_details': transaction_details})

@app.route('/transactions/<transaction_id>', methods=['DELETE'])
def delete_transaction_route(transaction_id):
    account_number = transaction_service.get_transaction(transaction_id)['account_number']
    transaction_list = delete_transaction(transaction_id)
    return jsonify({'transaction_list': transaction_list})

if __name__ == '__main__':
    app.run(debug=True)
