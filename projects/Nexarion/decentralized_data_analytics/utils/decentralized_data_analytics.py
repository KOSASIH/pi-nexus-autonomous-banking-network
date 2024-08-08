import pandas as pd
from utils.blockchain_utils import hash_data, send_transaction

def decentralized_data_analytics():
    # Load data
    data = pd.read_csv('processed_data.csv')

    # Hash data
    hashed_data = hash_data(data.to_json())

    # Send transaction to blockchain network
    tx_hash = send_transaction(hashed_data, 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID')

    print('Transaction Hash:', tx_hash)

if __name__ == '__main__':
    decentralized_data_analytics()
