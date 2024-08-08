import os
import json
import requests
from web3 import Web3, HTTPProvider
from eth_account import Account

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Set up Web3 provider
w3 = Web3(HTTPProvider(config['ethereum_node']))

# Set up Ethereum account
account = Account.from_key(config['private_key'])

# Set up Oracle Nexus contract
oracle_nexus_contract = w3.eth.contract(address=config['oracle_nexus_address'], abi=config['oracle_nexus_abi'])

# Function to fetch data from external API
def fetch_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to send data to Oracle Nexus contract
def send_data(data):
    # Encrypt data
    encrypted_data = oracle_nexus_contract.functions.encryptData(data).call()

    # Send transaction to Oracle Nexus contract
    tx_hash = oracle_nexus_contract.functions.sendRequest(encrypted_data).transact({'from': account.address, 'gas': 200000})

    # Wait for transaction to be mined
    w3.eth.waitForTransactionReceipt(tx_hash)

    print(f'Data sent to Oracle Nexus contract: {tx_hash.hex()}')

# Main script
if __name__ == '__main__':
    # Fetch data from external API
    data = fetch_data(config['api_url'])

    # Send data to Oracle Nexus contract
    if data:
        send_data(json.dumps(data).encode())
    else:
        print('Error fetching data')
