from flask import Flask, request, jsonify
from blockchain import EnergyTradingContract

app = Flask(__name__)

@app.route('/api/register-producer', methods=['POST'])
def register_producer():
    data = request.get_json()
    address = data['address']
    # Call register producer function
    EnergyTradingContract.registerProducer(address)
    return jsonify({'message': 'Producer registered successfully'})

@app.route('/api/register-consumer', methods=['POST'])
def register_consumer():
    data = request.get_json()
    address = data['address']
    # Call register consumer function
    EnergyTradingContract.registerConsumer(address)
    return jsonify({'message': 'Consumer registered successfully'})

@app.route('/api/trade-energy', methods=['POST'])
def trade_energy():
    data = request.get_json()
    producer = data['producer']
    consumer = data['consumer']
    amount = data['amount']
    # Call trade energy function
    EnergyTradingContract.tradeEnergy(producer, consumer, amount)
    return jsonify({'message': 'Energy traded successfully'})

if __name__ == '__main__':
    app.run(debug=True)
