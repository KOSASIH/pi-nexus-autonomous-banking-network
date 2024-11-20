from flask import Flask, jsonify, request, abort
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from src.auth.identity_management import IdentityManager
from src.blockchain.blockchain import Blockchain

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'  # Change this to a secure key
jwt = JWTManager(app)

# Initialize identity manager and blockchain
identity_manager = IdentityManager()
blockchain = Blockchain()

@app.route('/api/register', methods=['POST'])
def register_identity():
    """Register a new user identity."""
    data = request.json
    user_id = data.get('user_id')
    identity_data = data.get('identity_data')

    try:
        identity_manager.register_identity(user_id, identity_data)
        return jsonify({"message": "Identity registered successfully."}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/verify', methods=['GET'])
def verify_identity():
    """Verify a user identity."""
    user_id = request.args.get('user_id')
    exists = identity_manager.verify_identity(user_id)
    return jsonify({"user_id": user_id, "exists": exists}), 200

@app.route('/api/revoke', methods=['DELETE'])
@jwt_required()
def revoke_identity():
    """Revoke a user's identity."""
    user_id = request.json.get('user_id')
    try:
        identity_manager.revoke_identity(user_id)
        return jsonify({"message": "Identity revoked successfully."}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/transactions', methods=['POST'])
@jwt_required()
def create_transaction():
    """Create a new transaction."""
    data = request.json
    sender = data.get('sender')
    recipient = data.get('recipient')
    amount = data.get('amount')

    try:
        transaction_index = blockchain.new_transaction(sender, recipient, amount)
        return jsonify({"message": "Transaction created successfully.", "transaction_index": transaction_index}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/transactions', methods=['GET'])
@jwt_required()
def get_transactions():
    """Retrieve all transactions."""
    transactions = blockchain.current_transactions
    return jsonify({"transactions": transactions}), 200

@app.route('/api/login', methods=['POST'])
def login():
    """User  login to obtain JWT."""
    data = request.json
    user_id = data.get('user_id')
    # Here you would normally verify the user's credentials
    access_token = create_access_token(identity=user_id)
    return jsonify(access_token=access_token), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
