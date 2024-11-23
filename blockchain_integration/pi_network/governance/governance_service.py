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
with open('path/to/your/governance_contract_abi.json') as f:
    contract_abi = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)

@app.route('/api/governance/proposals', methods=['POST'])
def create_proposal():
    data = request.json
    description = data.get('description')
    proposer_address = data.get('proposer_address')

    tx = contract.functions.createProposal(description).buildTransaction({
        'from': proposer_address,
        'gas': 2000000,
        'gasPrice': w3.toWei('50', 'gwei'),
        'nonce': w3.eth.getTransactionCount(proposer_address),
    })

    private_key = os.getenv('PRIVATE_KEY')  # Ensure to keep your private key secure
    signed_tx = w3.eth.account.signTransaction(tx, private_key)
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    return jsonify({"transaction_hash": tx_hash.hex()}), 200

@app.route('/api/governance/vote', methods=['POST'])
def vote():
    data = request.json
    proposal_id = data.get('proposal_id')
    support = data.get('support')
    voter_address = data.get('voter_address')

    tx = contract.functions.vote(proposal_id, support).buildTransaction({
        'from': voter_address,
        'gas': 2000000,
        'gasPrice': w3.toWei('50', 'gwei'),
        'nonce': w3.eth.getTransactionCount(voter_address),
    })

    private_key = os.getenv('PRIVATE_KEY')  # Ensure to keep your private key secure
    signed_tx = w3.eth.account.signTransaction(tx, private_key)
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    return jsonify({"transaction_hash": tx_hash.hex()}), 200

@app.route('/api/governance/proposals/<int:proposal_id>', methods=['GET'])
def get_proposal(proposal_id):
    proposal = contract.functions.getProposal(proposal_id).call()
    return jsonify({
        "description": proposal[0],
        "proposer": proposal[1],
        "voteCountFor": proposal[2],
        "voteCountAgainst": proposal[3],
        "executed": proposal[4]
    }), 200

@app.route('/api/governance/proposal_count', methods=['GET'])
def get_proposal_count():
    count = contract.functions.getProposalCount().call()
    return jsonify({"proposal_count": count}), 200

if __name__ == '__main__':
    app.run(debug=True)
