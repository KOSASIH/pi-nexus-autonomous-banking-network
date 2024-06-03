import os
import json
from flask import Flask, request, jsonify
from web3 import Web3, HTTPProvider
from eth_account import Account
from eth_utils import to_checksum_address

app = Flask(__name__)

# Load environment variables
PI_NEXUS_CONTRACT_ADDRESS = os.environ['PI_NEXUS_CONTRACT_ADDRESS']
PI_NEXUS_ABI = os.environ['PI_NEXUS_ABI']
INFURA_PROJECT_ID = os.environ['INFURA_PROJECT_ID']
INFURA_PROJECT_SECRET = os.environ['INFURA_PROJECT_SECRET']

# Set up Web3 provider
w3 = Web3(HTTPProvider(f'https://mainnet.infura.io/v3/{INFURA_PROJECT_ID}'))

# Load PI Nexus contract ABI
with open(PI_NEXUS_ABI, 'r') as f:
    pi_nexus_abi = json.load(f)

# Set up PI Nexus contract instance
pi_nexus_contract = w3.eth.contract(address=to_checksum_address(PI_NEXUS_CONTRACT_ADDRESS), abi=pi_nexus_abi)

# Set up account for API to interact with contract
api_account = Account.from_key(os.environ['API_PRIVATE_KEY'])

@app.route('/api/v1/governance/proposals', methods=['GET'])
def get_governance_proposals():
    proposals = pi_nexus_contract.functions.getGovernanceProposals().call()
    return jsonify([{'id': proposal[0], 'description': proposal[1], 'votes': proposal[2]} for proposal in proposals])

@app.route('/api/v1/governance/proposals', methods=['POST'])
def create_governance_proposal():
    description = request.json['description']
    tx_hash = pi_nexus_contract.functions.createGovernanceProposal(description).transact({'from': api_account.address})
    return jsonify({'tx_hash': tx_hash})

@app.route('/api/v1/governance/proposals/<proposal_id>/vote', methods=['POST'])
def vote_on_governance_proposal(proposal_id):
    vote = request.json['vote']
    tx_hash = pi_nexus_contract.functions.voteOnGovernanceProposal(proposal_id, vote).transact({'from': api_account.address})
    return jsonify({'tx_hash': tx_hash})

@app.route('/api/v1/rewards', methods=['GET'])
def get_rewards():
    rewards = pi_nexus_contract.functions.getRewards().call()
    return jsonify([{'address': reward[0], 'amount': reward[1]} for reward in rewards])

@app.route('/api/v1/rewards', methods=['POST'])
def distribute_rewards():
    tx_hash = pi_nexus_contract.functions.distributeRewards().transact({'from': api_account.address})
    return jsonify({'tx_hash': tx_hash})

@app.route('/api/v1/liquidity', methods=['GET'])
def get_liquidity():
    liquidity = pi_nexus_contract.functions.getLiquidity().call()
    return jsonify({'amount': liquidity})

@app.route('/api/v1/liquidity', methods=['POST'])
def add_liquidity():
    amount = request.json['amount']
    tx_hash = pi_nexus_contract.functions.addLiquidity(amount).transact({'from': api_account.address})
    return jsonify({'tx_hash': tx_hash})

@app.route('/api/v1/borrowing', methods=['GET'])
def get_borrowing_requests():
    requests = pi_nexus_contract.functions.getBorrowingRequests().call()
    return jsonify([{'address': request[0], 'amount': request[1]} for request in requests])

@app.route('/api/v1/borrowing', methods=['POST'])
def request_borrowing():
    amount = request.json['amount']
    tx_hash = pi_nexus_contract.functions.requestBorrowing(amount).transact({'from': api_account.address})
    return jsonify({'tx_hash': tx_hash})

@app.route('/api/v1/borrowing/<request_id>/approve', methods=['POST'])
def approve_borrowing_request(request_id):
    tx_hash = pi_nexus_contract.functions.approveBorrowingRequest(request_id).transact({'from': api_account.address})
    return jsonify({'tx_hash': tx_hash})

if __name__ == '__main__':
    app.run(debug=True)
