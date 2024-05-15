# src/web/views.py
from flask import jsonify, request, Blueprint
from src.core.models import BankAccount

views = Blueprint('views', __name__)

@views.route('/api/bank_accounts', methods=['GET'])
def get_bank_accounts():
    # ...

@views.route('/api/bank_accounts/<int:account_id>', methods=['GET'])
def get_bank_account_by_id(account_id):
    # ...

@views.route('/api/bank_accounts', methods=['POST'])
def create_bank_account():
    # ...

@views.route('/api/bank_accounts/<int:account_id>', methods=['PUT'])
def update_bank_account(account_id):
    # ...

@views.route('/api/bank_accounts/<int:account_id>', methods=['DELETE'])
def delete_bank_account(account_id):
    # ...

@views.route('/api/bank_accounts/search', methods=['GET'])
def search_bank_accounts():
    # ...
