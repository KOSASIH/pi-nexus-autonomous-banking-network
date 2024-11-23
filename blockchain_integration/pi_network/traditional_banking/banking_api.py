from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Configuration for bank API integration
BANK_API_URL = os.getenv('BANK_API_URL', 'https://api.yourbank.com')
API_KEY = os.getenv('API_KEY', 'YOUR_API_KEY')

@app.route('/api/banking/accounts', methods=['GET'])
def get_accounts():
    """Fetches the list of accounts for the authenticated user."""
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    response = requests.get(f'{BANK_API_URL}/accounts', headers=headers)
    
    if response.status_code == 200:
        return jsonify(response.json()), 200
    else:
        return jsonify({"error": response.json()}), response.status_code

@app.route('/api/banking/transfer', methods=['POST'])
def transfer():
    """Transfers money between accounts."""
    data = request.json
    from_account = data.get('from_account')
    to_account = data.get('to_account')
    amount = data.get('amount')

    if not from_account or not to_account or not amount:
        return jsonify({"error": "Missing required fields"}), 400

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount
    }
    
    response = requests.post(f'{BANK_API_URL}/transfer', json=payload, headers=headers)

    if response.status_code == 200:
        return jsonify({"message": "Transfer successful", "transaction_id": response.json().get('transaction_id')}), 200
    else:
        return jsonify({"error": response.json()}), response.status_code

@app.route('/api/banking/transaction_history/<account_id>', methods=['GET'])
def get_transaction_history(account_id):
    """Fetches the transaction history for a specific account."""
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    response = requests.get(f'{BANK_API_URL}/accounts/{account_id}/transactions', headers=headers)

    if response.status_code == 200:
        return jsonify(response.json()), 200
    else:
        return jsonify({"error": response.json()}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)
