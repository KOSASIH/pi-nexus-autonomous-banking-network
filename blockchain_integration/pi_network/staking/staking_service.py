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
with open('path/to/your/staking_contract_abi.json') as f:
    contract_abi = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)

@app.route('/api/staking/stake', methods=['POST'])
def stake():
    data = request.json
    amount = data.get('amount')
    staker_address = data.get('staker_address')

    tx = contract.functions.stake(amount).buildTransaction({
        'from': staker_address,
        'gas': 2000000,
        'gasPrice': w3.toWei('50', 'gwei'),
        'nonce': w3.eth.getTransactionCount(staker_address),
    })

    private_key = os.getenv('PRIVATE_KEY')  # Ensure to keep your private key secure
    signed_tx = w3.eth.account.signTransaction(tx, private_key)
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    return jsonify({"transaction_hash": tx_hash.hex()}), 200

@app.route('/api/staking/unstake', methods=['POST'])
def unstake():
    data = request.json
    amount = data.get('amount')
    staker_address = data.get('staker_address')

    tx = contract.functions.unstake(amount).buildTransaction({
        'from': staker_address,
        'gas': 2000000,
        'gasPrice': w3.toWei('50', 'gwei'),
        'nonce': w3.eth.getTransactionCount(staker_address),
    })

    private_key = os.getenv('PRIVATE_KEY')  # Ensure to keep your private key secure
    signed_tx = w3.eth.account.signTransaction(tx, private_key)
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    return jsonify({"transaction_hash": tx_hash.hex()}), 200

@app.route('/api/staking/claim_reward', methods=['POST'])
def claim_reward():
    data = request.json
    staker_address = data.get('staker_address')

    tx = contract.functions.claimReward().buildTransaction({
        'from': staker_address,
        'gas': 2000000,
        'gasPrice': w3.toWei('50', 'gwei'),
        'nonce': w3.eth.getTransactionCount(staker_address),
    })

    private_key = os.getenv('PRIVATE_KEY')  # Ensure to keep your private key secure
    signed_tx = w3.eth.account.signTransaction(tx, private_key)
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    return jsonify({"transaction_hash": tx_hash.hex()}), 200

@app.route('/api/staking/stake_info/<address>', methods=['GET'])
def get_stake_info(address):
    stake_amount = contract.functions.getStake(address).call()
    reward_amount = contract.functions.getReward(address).call()
    return jsonify({
        "stake_amount": stake_amount,
        "reward_amount": reward_amount
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
