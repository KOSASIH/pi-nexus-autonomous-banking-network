import json
from pi_network.api.models import User, Account, Transaction

def load_fixtures():
    # Load the test fixtures from JSON files
    with open('fixtures/users.json', 'r') as f:
        users = json.load(f)
    with open('fixtures/accounts.json', 'r') as f:
        accounts = json.load(f)
    with open('fixtures/transactions.json', 'r') as f:
        transactions = json.load(f)

    # Create the test data
    for user in users:
        User.create(**user)
    for account in accounts:
        Account.create(**account)
    for transaction in transactions:
        Transaction.create(**transaction)
