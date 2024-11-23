from flask import Flask, request, jsonify
from web3 import Web3
import json
import os

app = Flask(__name__)

# Connect to Ethereum node
ETH_NODE_URL = os.getenv('ETH_NODE_URL', 'https://your.ethereum.node')
w3 = Web3(Web3.HTTPProvider(ETH_NODE_URL))

# Load smart contract ABI and address
CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS', 'YOUR_CONTRACT_ADDRESS')
with open('path/to/your/contract_abi.json') as f:
    contract_abi = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)

# Sample IoT device data
iot_devices = {
    "device1": {"owner": "0xYourEthereumAddress1", "balance": 0},
    "device2": {"owner": "0xYourEthereumAddress2", "balance": 0},
}

@app.route('/api/iot/payment', methods=['POST'])
def make_payment():
    data = request.json
    device_id = data.get('device_id')
    amount = data.get('amount')
    payer_address = data.get('payer_address')

    if device_id not in iot_devices:
        return jsonify({"error": "Device not found"}), 404

    device = iot_devices[device_id]
    if w3.toWei(amount, 'ether') > w3.eth.get_balance(payer_address):
        return jsonify({"error": "Insufficient funds"}), 400

    # Create and send transaction
    tx = contract.functions.makePayment(device['owner'], w3.toWei(amount, 'ether')).buildTransaction({
        'from': payer_address,
        'value': w3.toWei(amount, 'ether'),
        'gas': 2000000,
        'gasPrice': w3.toWei('50', 'gwei'),
        'nonce': w3.eth.getTransactionCount(payer_address),
    })

    # Sign and send the transaction
    private_key = os.getenv('PRIVATE_KEY')  # Ensure to keep your private key secure
    signed_tx = w3.eth.account.signTransaction(tx, private_key)
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    return jsonify({"transaction_hash": tx_hash.hex()}), 200

@app.route('/api/iot/devices', methods=['GET'])
def get_devices():
    return jsonify(iot_devices), 200

if __name__ == '__main__':
    app.run(debug=True)
