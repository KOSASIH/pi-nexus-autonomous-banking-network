from flask import Blueprint, request, jsonify
from models.account import Account

account_blueprint = Blueprint('account', __name__)

@account_blueprint.route('/accounts', methods=['GET'])
def get_accounts():
    # Implement getting all accounts logic
    pass

@account_blueprint.route('/accounts/<account_number>', methods=['GET'])
def get_account(account_number):
    # Implement getting a single account logic
    pass
