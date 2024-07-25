# Importing necessary libraries
import pandas as pd
from web3 import Web3

# Class for blockchain analytics
class PiNetworkBlockchainAnalytics:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))

    # Function to get blockchain data
    def get_blockchain_data(self):
        block_number = self.w3.eth.block_number
        block_data = self.w3.eth.get_block(block_number)
        return block_data

    # Function to analyze blockchain data
    def analyze_blockchain_data(self, data):
        transactions = data['transactions']
        transaction_values = [tx['value'] for tx in transactions]
        return transaction_values

# Example usage
ba = PiNetworkBlockchainAnalytics()
data = ba.get_blockchain_data()
transaction_values = ba.analyze_blockchain_data(data)
print(transaction_values)
