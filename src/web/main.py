# src/web/main.py
from flask import Flask, jsonify
from src.core.models import BankAccount

app = Flask(__name__)

@app.route('/api/bank_accounts', methods=['GET'])
def get_bank_accounts():
    bank_accounts = BankAccount.query.all()
    return jsonify([ba.to_dict() for ba in bank_accounts])

if __name__ == '__main__':
    app.run(debug=True)
