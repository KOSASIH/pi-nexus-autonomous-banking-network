from flask import Flask, render_template, request
from blockchain_integration.pi_token import PiToken

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_transaction', methods=['POST'])
def send_transaction():
    from_user_id = request.form['from_user_id']
    to_user_id = request.form['to_user_id']
    amount = request.form['amount']
    pi_token = PiToken()
    tx_hash = pi_token.send_transaction(from_user_id, to_user_id, amount)
    return render_template('transaction_sent.html', tx_hash=tx_hash)

if __name__ == '__main__':
    app.run(debug=True)
