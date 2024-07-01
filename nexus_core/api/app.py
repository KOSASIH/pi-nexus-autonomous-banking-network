from flask import Flask, request, jsonify
from nexus_core.blockchain import Blockchain

app = Flask(__name__)

blockchain = Blockchain()

@app.route("/transactions", methods=["GET"])
def get_transactions():
    transactions = blockchain.get_transactions()
    return jsonify(transactions)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    predictions = blockchain.predict(data)
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)
