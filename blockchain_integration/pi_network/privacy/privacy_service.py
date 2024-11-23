import json
from flask import Flask, request, jsonify
from web3 import Web3

app = Flask(__name__)

# Connect to Ethereum network
w3 = Web3(Web3.HTTPProvider('https://your.ethereum.node'))

# Smart contract ABI and address
contract_address = '0xYourContractAddress'
contract_abi = json.loads('''[
    {
        "constant": true,
        "inputs": [{"name": "_proofId", "type": "bytes32"}],
        "name": "getProofDetails",
        "outputs": [
            {"name": "", "type": "bytes32"},
            {"name": "", "type": "bytes32"},
            {"name": "", "type": "address"}
        ],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {"name": "_proofId", "type": "bytes32"},
            {"name": "_commitment", "type": "bytes32"},
            {"name": "_proofData", "type": "bytes32"}
        ],
        "name": "createProof",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    }
]''')

contract = w3.eth.contract(address=contract_address, abi=contract_abi)

@app.route('/create_proof', methods=['POST'])
def create_proof():
    data = request.json
    proof_id = data['proofId']
    commitment = data['commitment']
    proof_data = data['proofData']

    # Call the smart contract function
    tx_hash = contract.functions.createProof(proof_id, commitment, proof_data).transact()
    return jsonify({'transaction_hash': tx_hash.hex()})

@app.route('/get_proof/<proof_id>', methods=['GET'])
def get_proof(proof_id):
    proof_details = contract.functions.getProofDetails(proof_id).call()
    return jsonify({
        'commitment': proof_details[0],
        'proof_data': proof_details[1],
        'verifier': proof_details[2]
    })

if __name__ == '__main__':
    app.run(debug=True)
