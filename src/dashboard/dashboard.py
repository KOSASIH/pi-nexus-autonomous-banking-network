from flask import Flask, render_template, request, redirect, url_for, flash
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.auth.identity_management import IdentityManager
from src.blockchain.blockchain import Blockchain

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key

# Initialize identity manager and blockchain
identity_manager = IdentityManager()
blockchain = Blockchain()

@app.route('/dashboard')
@jwt_required()
def dashboard():
    """Render the main dashboard."""
    user_id = get_jwt_identity()
    transactions = blockchain.current_transactions  # Get current transactions
    return render_template('dashboard.html', user_id=user_id, transactions=transactions)

@app.route('/dashboard/register', methods=['POST'])
@jwt_required()
def register_identity():
    """Register a new user identity."""
    user_id = get_jwt_identity()
    identity_data = request.form.get('identity_data')

    try:
        identity_manager.register_identity(user_id, identity_data)
        flash("Identity registered successfully.", "success")
    except ValueError as e:
        flash(str(e), "error")
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard/verify', methods=['POST'])
@jwt_required()
def verify_identity():
    """Verify a user identity."""
    user_id = request.form.get('user_id')
    exists = identity_manager.verify_identity(user_id)
    flash(f"Identity verification for {user_id}: {'Exists' if exists else 'Does not exist'}", "info")
    return redirect(url_for('dashboard'))

@app.route('/dashboard/revoke', methods=['POST'])
@jwt_required()
def revoke_identity():
    """Revoke a user's identity."""
    user_id = request.form.get('user_id')
    try:
        identity_manager.revoke_identity(user_id)
        flash("Identity revoked successfully.", "success")
    except ValueError as e:
        flash(str(e), "error")
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard/transactions', methods=['POST'])
@jwt_required()
def create_transaction():
    """Create a new transaction."""
    sender = get_jwt_identity()
    recipient = request.form.get('recipient')
    amount = float(request.form.get('amount'))

    try:
        transaction_index = blockchain.new_transaction(sender, recipient, amount)
        flash(f"Transaction created successfully. Index: {transaction_index}", "success")
    except Exception as e:
        flash(str(e), "error")
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard/notifications')
@jwt_required()
def notifications():
    """Render notifications for the user."""
    # Placeholder for notifications logic
    return render_template('notifications.html')

if __name__ == '__main__':
    app.run(debug=True)
