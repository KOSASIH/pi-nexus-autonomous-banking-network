import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from web3 import Web3, HTTPProvider
from data_sharing_contract import DataSharingContract

app = Flask(__name__)
CORS(app)

contract_address = '0x...contract address...'
abi = json.load(open('data_sharing_contract.abi', 'r'))
provider_url = 'https://mainnet.infura.io/v3/...project id...'
contract = DataSharingContract(contract_address, abi, provider_url)

@app.route('/register_provider', methods=['POST'])
def register_provider():
    provider_address = request.json['provider_address']
    data_hash = request.json['data_hash']
    contract.register_data_provider(provider_address, data_hash)
    return jsonify({'message': 'Provider registered successfully'})

@app.route('/register_consumer', methods=['POST'])
def register_consumer():
    consumer_address = request.json['consumer_address']
    contract.register_data_consumer(consumer_address)
    return jsonify({'message': 'Consumer registered successfully'})

@app.route('/request_data', methods=['POST'])
def request_data():
    consumer_address = request.json['consumer_address']
    data_hash = request.json['data_hash']
    contract.request_data(consumer_address, data_hash)
    return jsonify({'message': 'Data request sent successfully'})

@app.route('/provide_data', methods=['POST'])
def provide_data():
    provider_address = request.json['provider_address']
    data_hash = request.json['data_hash']
    data = request.json['data']
    contract.provide_data(provider_address, data_hash, data)
    return jsonify({'message': 'Data provided successfully'})

@app.route('/get_data', methods=['GET'])
def get_data():
    data_hash = request.args.get('data_hash')
    data = contract.get_data(data_hash)
    return jsonify({'data': data})

@app.route('/get_providers', methods=['GET'])
def get_providers():
    providers = contract.get_data_providers()
    return jsonify({'providers': providers})

@app.route('/get_consumers', methods=['GET'])
def get_consumers():
    consumers = contract.get_data_consumers()
    return jsonify({'consumers': consumers})

if __name__ == '__main__':
    app.run(debug=True)
