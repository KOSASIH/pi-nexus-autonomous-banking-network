from flask import Flask, jsonify
from blockchain import BlockchainService
from transaction import TransactionService
from user import UserService

app = Flask(__name__)

blockchain_service = BlockchainService()
transaction_service = TransactionService()
user_service = UserService()

@app.route('/blockchain', methods=['GET'])
def get_blockchain():
  return jsonify(blockchain_service.get_blockchain())

@app.route('/transaction', methods=['GET'])
def get_transaction():
  return jsonify(transaction_service.get_transactions())

@app.route('/user', methods=['GET'])
def get_user():
  return jsonify(user_service.get_users())

if __name__ == '__main__':
  app.run(debug=True)
