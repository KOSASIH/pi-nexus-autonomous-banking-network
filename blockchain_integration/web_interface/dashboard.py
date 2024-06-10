from flask import Flask, render_template, request, redirect, url_for
from blockchain_integration.pi_token import PiToken
from blockchain_integration.pi_network_api import PiNetworkAPI

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    pi_token = PiToken()
    balance = pi_token.get_token_balance('0xYourAddress')
    return render_template('dashboard.html', balance=balance)

@app.route('/send_transaction', methods=['POST'])
def send_transaction():
    from_user_id = request.form['from_user_id']
    to_user_id = request.form['to_user_id']
    amount = request.form['amount']
    pi_token = PiToken()
    tx_hash = pi_token.send_transaction(from_user_id, to_user_id, amount)
    return redirect(url_for('transaction_sent', tx_hash=tx_hash))

@app.route('/transaction_sent')
def transaction_sent():
    tx_hash = request.args.get('tx_hash')
    return render_template('transaction_sent.html', tx_hash=tx_hash)

@app.route('/freeze_token', methods=['POST'])
def freeze_token():
    user_id = request.form['user_id']
    amount = request.form['amount']
    pi_token = PiToken()
    pi_token.freeze_token(user_id, amount)
    return redirect(url_for('token_frozen'))

@app.route('/token_frozen')
def token_frozen():
    return render_template('token_frozen.html')

@app.route('/unfreeze_token', methods=['POST'])
def unfreeze_token():
    user_id = request.form['user_id']
    amount = request.form['amount']
    pi_token = PiToken()
    pi_token.unfreeze_token(user_id, amount)
    return redirect(url_for('token_unfrozen'))

@app.route('/token_unfrozen')
def token_unfrozen():
    return render_template('token_unfrozen.html')

@app.route('/vote_token', methods=['POST'])
def vote_token():
    user_id = request.form['user_id']
    amount = request.form['amount']
    pi_token = PiToken()
    pi_token.vote_token(user_id, amount)
    return redirect(url_for('token_voted'))

@app.route('/token_voted')
def token_voted():
    return render_template('token_voted.html')

@app.route('/delegate_token', methods=['POST'])
def delegate_token():
    user_id = request.form['user_id']
    delegatee = request.form['delegatee']
    amount = request.form['amount']
    pi_token = PiToken()
    pi_token.delegate_token(user_id, delegatee, amount)
    return redirect(url_for('token_delegated'))

@app.route('/token_delegated')
def token_delegated():
    return render_template('token_delegated.html')

@app.route('/stake_token', methods=['POST'])
def stake_token():
    user_id = request.form['user_id']
    amount = request.form['amount']
    pi_token = PiToken()
    pi_token.stake_token(user_id, amount)
    return redirect(url_for('token_staked'))

@app.route('/token_staked')
def token_staked():
    return render_template('token_staked.html')

if __name__ == '__main__':
    app.run(debug=True)
